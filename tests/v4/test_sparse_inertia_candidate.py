"""Standalone C++ differential tests of the actual generated loop replacements."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/v4'))
import build_sparse_inertia_candidate as candidate

SOURCE = ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/inertia_controller.h'


class SparseInertiaCandidateTests(unittest.TestCase):
    def test_bound_header_and_unchanged_frozen_file(self):
        before = SOURCE.read_bytes()
        code, manifest = candidate.render_header(SOURCE)
        self.assertEqual(before, SOURCE.read_bytes())
        self.assertIn(candidate.MASS_NEW, code)
        self.assertIn(candidate.EFFORT_NEW, code)
        self.assertNotIn(candidate.MASS_OLD, code)
        self.assertFalse(manifest['physics_frequency_changed'])
        self.assertFalse(manifest['default_configuration_changed'])
        self.assertIn('std::abs(out.force[i])>1e6', code)

    def test_changed_baseline_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'changed.h'
            path.write_bytes(SOURCE.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError, 'sha256'):
                candidate.render_header(path)

    def test_cpp_projection_equivalence_and_nonfinite_guards(self):
        prefix = r'''
#include <array>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>
#include <limits>
#include <iostream>
struct Model {int nv;};
struct Result {std::array<double,16> mass{};std::array<double,4> acceleration_force{},velocity_bias{};};
'''
        signature = '(Model* m,const std::vector<double>& T,const std::vector<double>& M,const std::vector<double>& a,const std::vector<double>& curve,const std::vector<double>& bias,const std::vector<double>& zero_bias){Result out;\n'
        tail = '''
      for(int w=0;w<m->nv;++w){inertia+=M[u*m->nv+w]*a[w];centripetal+=M[u*m->nv+w]*curve[w];}
      out.acceleration_force[i]+=T[u*4+i]*inertia;
      out.velocity_bias[i]+=T[u*4+i]*centripetal;
    }
    return out;
}
'''
        code = prefix
        code += 'Result baseline'+signature+candidate.MASS_OLD+'\n'+candidate.EFFORT_OLD+tail
        code += 'Result optimized'+signature+candidate.MASS_NEW+'\n'+candidate.EFFORT_NEW+tail
        code += r'''
int main(){
  std::mt19937 gen(107);std::uniform_real_distribution<double> dist(-2,2);
  int cases=0;
  for(int n:{18,24,300,408})for(int trial=0;trial<12;++trial){
    Model m{n};std::vector<double>T(n*4),M(n*n),a(n),curve(n),bias(n),zero(n);
    for(auto&v:M)v=dist(gen);
    for(int u=0;u<n;++u){a[u]=dist(gen);curve[u]=dist(gen);bias[u]=dist(gen);zero[u]=dist(gen);
      for(int i=0;i<4;++i)if(trial==0||((u*7+i*11+trial)%13==0))T[u*4+i]=dist(gen);
    }
    // Include exact negative zero and a nonzero smaller than any useful tolerance.
    T[0]=-0.0;T[1]=1e-100;
    auto b=baseline(&m,T,M,a,curve,bias,zero),o=optimized(&m,T,M,a,curve,bias,zero);
    if(b.mass!=o.mass||b.acceleration_force!=o.acceleration_force||b.velocity_bias!=o.velocity_bias)return 2;
    ++cases;
  }
  Model m{18};std::vector<double>T(72),M(324,1),a(18),curve(18),bias(18),zero(18);
  int guards=0;
  for(auto*values:{&M,&T,&a,&curve,&bias,&zero}){
    auto old=(*values)[0];(*values)[0]=std::numeric_limits<double>::quiet_NaN();
    try{optimized(&m,T,M,a,curve,bias,zero);}catch(const std::runtime_error&){++guards;}
    (*values)[0]=old;
  }
  std::cout<<"exact_cases="<<cases<<" nonfinite_guards="<<guards<<"\n";
  return cases==48&&guards==6?0:3;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source, binary = Path(directory)/'test.cc', Path(directory)/'test'
            source.write_text(code)
            subprocess.run(['g++','-O2','-std=c++17','-Wall','-Wextra',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            result = subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('exact_cases=48 nonfinite_guards=6', result.stdout)


if __name__ == '__main__':
    unittest.main()

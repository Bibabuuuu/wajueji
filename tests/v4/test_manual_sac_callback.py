from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from run_manual_residual_sac import Chunk


class TerminalCallbackTests(unittest.TestCase):
    def callback(self, remaining, done, steps=1, minimum=10):
        directory=TemporaryDirectory();self.addCleanup(directory.cleanup)
        cb=Chunk(SimpleNamespace(remaining=lambda:remaining),Path(directory.name)/'progress.json',minimum,600)
        cb.model=SimpleNamespace(num_timesteps=steps,_n_updates=0)
        cb.started_steps=0;cb.last=time.monotonic();cb.locals={'dones':[done]}
        return cb

    def test_reserve_reached_on_terminal_stores_then_stops(self):
        cb=self.callback(599,True)
        self.assertTrue(cb._on_step())
        self.assertTrue(cb.stop_next)
        self.assertFalse(cb._on_step())

    def test_reserve_reached_mid_episode_stops_without_fake_terminal(self):
        cb=self.callback(599,False)
        self.assertFalse(cb._on_step());self.assertFalse(cb.stop_next)

    def test_normal_completed_chunk_retains_terminal(self):
        cb=self.callback(1000,True,steps=10)
        self.assertTrue(cb._on_step());self.assertFalse(cb._on_step())


if __name__=='__main__':unittest.main()

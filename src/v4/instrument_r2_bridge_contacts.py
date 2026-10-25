"""Derive a bookkeeping-only R2 bridge with per-rock contact counters."""

from __future__ import annotations


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"per_rock_contact_anchor_{label}:{count}")
    return source.replace(old, new)


def instrument_per_rock_contacts(source: str) -> str:
    """Add per-rock ground and bucket counters without changing dynamics."""
    source = _replace_once(
        source,
        "    int forbidden_contact_count = 0;\n",
        "    int forbidden_contact_count = 0;\n"
        "    std::vector<int> rock_ground_contact_substeps_by_rock(rock_count);\n"
        "    std::vector<int> rock_bucket_contact_substeps_by_rock(rock_count);\n",
        "declarations")
    source = _replace_once(
        source,
        "      std::cout << \",\\\"contact_substeps\\\":{\\\"rock_ground\\\":\"\n"
        "                << rock_ground_contact_substeps << \",\\\"rock_bucket\\\":\"\n"
        "                << rock_bucket_contact_substeps << \",\\\"rock_rock\\\":\"\n"
        "                << rock_rock_contact_substeps << \",\\\"rock_truck\\\":\"\n"
        "                << rock_truck_contact_substeps << \",\\\"rock_ramp\\\":\"\n"
        "                << rock_ramp_contact_substeps << \"}\"\n"
        "                << \",\\\"forbidden_contact_count\\\":\" << forbidden_contact_count\n",
        "      std::cout << \",\\\"contact_substeps\\\":{\\\"rock_ground\\\":\"\n"
        "                << rock_ground_contact_substeps << \",\\\"rock_bucket\\\":\"\n"
        "                << rock_bucket_contact_substeps << \",\\\"rock_rock\\\":\"\n"
        "                << rock_rock_contact_substeps << \",\\\"rock_truck\\\":\"\n"
        "                << rock_truck_contact_substeps << \",\\\"rock_ramp\\\":\"\n"
        "                << rock_ramp_contact_substeps << \"}\";\n"
        "      array(\"rock_ground_contact_substeps_by_rock\",\n"
        "            rock_ground_contact_substeps_by_rock.data(), rock_count);\n"
        "      array(\"rock_bucket_contact_substeps_by_rock\",\n"
        "            rock_bucket_contact_substeps_by_rock.data(), rock_count);\n"
        "      std::cout << \",\\\"forbidden_contact_count\\\":\" << forbidden_contact_count\n",
        "emit")
    source = _replace_once(
        source,
        "      forbidden_contact_count = 0;\n"
        "      bucket_rock_impulse.fill(0);\n",
        "      forbidden_contact_count = 0;\n"
        "      std::fill(rock_ground_contact_substeps_by_rock.begin(),\n"
        "                rock_ground_contact_substeps_by_rock.end(), 0);\n"
        "      std::fill(rock_bucket_contact_substeps_by_rock.begin(),\n"
        "                rock_bucket_contact_substeps_by_rock.end(), 0);\n"
        "      bucket_rock_impulse.fill(0);\n",
        "reset")
    source = _replace_once(
        source,
        "        bool truck_this_substep = false;\n"
        "        std::array<double, 3> substep_bucket_force{};\n",
        "        bool truck_this_substep = false;\n"
        "        std::vector<bool> ground_by_rock(rock_count, false);\n"
        "        std::vector<bool> bucket_by_rock(rock_count, false);\n"
        "        std::array<double, 3> substep_bucket_force{};\n",
        "flags")
    source = _replace_once(
        source,
        "          const bool rock_is_first = first_rock >= 0;\n"
        "          const int other_geom = rock_is_first ? contact.geom2 : contact.geom1;\n",
        "          const bool rock_is_first = first_rock >= 0;\n"
        "          const int rock_index = rock_is_first ? first_rock : second_rock;\n"
        "          const int other_geom = rock_is_first ? contact.geom2 : contact.geom1;\n",
        "rock_index")
    source = _replace_once(
        source,
        "            bucket_this_substep = true;\n"
        "            mjtNum local[6] = {}, world_force[3] = {}, world_torque[3] = {};\n",
        "            bucket_this_substep = true;\n"
        "            bucket_by_rock[rock_index] = true;\n"
        "            mjtNum local[6] = {}, world_force[3] = {}, world_torque[3] = {};\n",
        "bucket")
    source = _replace_once(
        source,
        "          } else if (other_name && std::string(other_name) == \"ground_plane\") {\n"
        "            ground_this_substep = true;\n"
        "          } else if (StartsWith(other_name, \"r2_surface_\")) {\n"
        "            ground_this_substep = true;\n"
        "          } else if (StartsWith(other_name, \"r2_ground_\")) {\n"
        "            ground_this_substep = true;\n",
        "          } else if (other_name && std::string(other_name) == \"ground_plane\") {\n"
        "            ground_this_substep = true;\n"
        "            ground_by_rock[rock_index] = true;\n"
        "          } else if (StartsWith(other_name, \"r2_surface_\")) {\n"
        "            ground_this_substep = true;\n"
        "            ground_by_rock[rock_index] = true;\n"
        "          } else if (StartsWith(other_name, \"r2_ground_\")) {\n"
        "            ground_this_substep = true;\n"
        "            ground_by_rock[rock_index] = true;\n",
        "ground")
    source = _replace_once(
        source,
        "        rock_ground_contact_substeps += ground_this_substep;\n",
        "        for (int index = 0; index < rock_count; ++index) {\n"
        "          rock_ground_contact_substeps_by_rock[index] += ground_by_rock[index];\n"
        "          rock_bucket_contact_substeps_by_rock[index] += bucket_by_rock[index];\n"
        "        }\n"
        "        rock_ground_contact_substeps += ground_this_substep;\n",
        "accumulate")
    return source

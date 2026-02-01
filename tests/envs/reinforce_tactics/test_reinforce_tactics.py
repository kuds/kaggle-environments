"""Tests for the Reinforce Tactics Kaggle environment."""

from kaggle_environments import evaluate, make

env = None


def before_each(configuration=None):
    global env
    env = make(
        "reinforce_tactics",
        configuration={"mapSeed": 42, **(configuration or {})},
        debug=True,
    )


# ---------------------------------------------------------------------------
# Basic environment setup
# ---------------------------------------------------------------------------


def test_has_correct_timeouts():
    before_each()
    assert env.configuration.actTimeout == 5
    assert env.configuration.runTimeout == 1200


def test_has_correct_defaults():
    before_each()
    assert env.configuration.mapWidth == 20
    assert env.configuration.mapHeight == 20
    assert env.configuration.startingGold == 250
    assert env.configuration.fogOfWar is False
    assert env.configuration.enabledUnits == "W,M,C,A,K,R,S,B"


def test_to_json():
    before_each()
    json = env.toJSON()
    assert json["name"] == "reinforce_tactics"
    assert json["statuses"] == ["ACTIVE", "INACTIVE"]


def test_can_reset():
    before_each()
    state = env.reset()
    assert len(state) == 2
    assert state[0]["status"] == "ACTIVE"
    assert state[1]["status"] == "INACTIVE"


def test_initial_observations():
    before_each()
    state = env.reset()
    obs = state[0]["observation"]
    assert obs["mapWidth"] == 20
    assert obs["mapHeight"] == 20
    assert obs["turnNumber"] == 0
    assert obs["player"] == 0
    assert len(obs["board"]) == 20
    assert len(obs["board"][0]) == 20
    assert obs["gold"] == [250, 250]

    obs2 = state[1]["observation"]
    assert obs2["player"] == 1


def test_board_has_grass_tiles():
    """Map generation should produce mostly grass tiles, not ocean."""
    before_each()
    state = env.reset()
    board = state[0]["observation"]["board"]
    grass_count = sum(row.count("p") for row in board)
    # Most tiles should be grass (at least 50% of the 400 tiles)
    assert grass_count > 200, f"Expected mostly grass tiles, got {grass_count} out of 400"


def test_board_has_structures():
    """Map should have HQ and buildings for both players."""
    before_each()
    state = env.reset()
    structures = state[0]["observation"]["structures"]
    # Should have at least 2 HQs and 4 buildings
    hq_count = sum(1 for s in structures if s["type"] == "h")
    building_count = sum(1 for s in structures if s["type"] == "b")
    assert hq_count == 2, f"Expected 2 HQs, got {hq_count}"
    assert building_count >= 4, f"Expected at least 4 buildings, got {building_count}"


def test_board_has_towers():
    """Map should have neutral towers in the centre."""
    before_each()
    state = env.reset()
    structures = state[0]["observation"]["structures"]
    tower_count = sum(1 for s in structures if s["type"] == "t")
    assert tower_count > 0, "Expected neutral towers on the map"


# ---------------------------------------------------------------------------
# Turn mechanics
# ---------------------------------------------------------------------------


def test_end_turn_swaps_active_player():
    before_each()
    env.reset()
    state = env.step([[{"type": "end_turn"}], None])
    assert state[0]["status"] == "INACTIVE"
    assert state[1]["status"] == "ACTIVE"


def test_end_turn_and_back():
    before_each()
    env.reset()
    env.step([[{"type": "end_turn"}], None])
    state = env.step([None, [{"type": "end_turn"}]])
    assert state[0]["status"] == "ACTIVE"
    assert state[1]["status"] == "INACTIVE"


# ---------------------------------------------------------------------------
# Unit creation
# ---------------------------------------------------------------------------


def test_create_unit_at_building():
    before_each()
    state = env.reset()
    structures = state[0]["observation"]["structures"]
    # Find a building owned by player 1
    p1_building = next(
        s for s in structures if s["owner"] == 1 and s["type"] == "b"
    )

    actions = [
        {
            "type": "create_unit",
            "unit_type": "W",
            "x": p1_building["x"],
            "y": p1_building["y"],
        },
        {"type": "end_turn"},
    ]
    state = env.step([actions, None])

    # Player 1 should now have a warrior
    units = state[0]["observation"]["units"]
    warriors = [u for u in units if u["type"] == "W" and u["owner"] == 1]
    assert len(warriors) == 1
    assert warriors[0]["x"] == p1_building["x"]
    assert warriors[0]["y"] == p1_building["y"]

    # Gold should be reduced by warrior cost (200)
    gold = state[0]["observation"]["gold"]
    # After creating a warrior (200g) and receiving income, gold should differ
    assert gold[0] != 250  # Gold has changed


def test_create_unit_deducts_gold():
    before_each()
    state = env.reset()
    initial_gold = state[0]["observation"]["gold"][0]

    structures = state[0]["observation"]["structures"]
    p1_building = next(
        s for s in structures if s["owner"] == 1 and s["type"] == "b"
    )

    actions = [
        {
            "type": "create_unit",
            "unit_type": "W",
            "x": p1_building["x"],
            "y": p1_building["y"],
        },
        {"type": "end_turn"},
    ]
    state = env.step([actions, None])
    new_gold = state[0]["observation"]["gold"][0]

    # Warrior costs 200 gold. Income is collected at the start of the
    # NEXT player's turn (not the ending player's), so P1 gold should
    # simply be initial - 200.
    expected = initial_gold - 200
    assert new_gold == expected, f"Expected {expected}, got {new_gold}"


# ---------------------------------------------------------------------------
# Invalid actions
# ---------------------------------------------------------------------------


def test_invalid_action_loses():
    """An invalid action dict should cause the agent to lose."""
    before_each()
    env.reset()
    state = env.step([["not_a_dict"], None])
    assert state[0]["status"] == "DONE"
    assert state[0]["reward"] == -1
    assert state[1]["reward"] == 1


def test_unknown_action_type_loses():
    """An action with an unknown type should cause the agent to lose."""
    before_each()
    env.reset()
    state = env.step([[{"type": "fly_to_moon"}], None])
    assert state[0]["status"] == "DONE"
    assert state[0]["reward"] == -1
    assert state[1]["reward"] == 1


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_can_render_ansi():
    before_each()
    env.reset()
    out = env.render(mode="ansi")
    assert "Turn" in out
    assert "P1 Gold" in out
    assert "P2 Gold" in out


# ---------------------------------------------------------------------------
# Built-in agents
# ---------------------------------------------------------------------------


def test_can_run_random_agent():
    before_each()
    env.run(["random", "random"])
    assert env.done


def test_can_run_aggressive_agent():
    before_each()
    env.run(["aggressive", "aggressive"])
    assert env.done


def test_can_run_mixed_agents():
    before_each()
    env.run(["random", "aggressive"])
    assert env.done


def test_rewards_sum_to_zero_or_both_zero():
    """Rewards should be +1/-1 (win/loss) or both 0 (draw)."""
    before_each()
    result = env.run(["random", "aggressive"])
    final = result[-1]
    r0 = final[0]["reward"]
    r1 = final[1]["reward"]
    assert r0 + r1 == 0 or (r0 == 0 and r1 == 0)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def test_can_evaluate():
    rewards = evaluate(
        "reinforce_tactics",
        ["random", "random"],
        num_episodes=2,
        configuration={"mapSeed": 42, "episodeSteps": 10},
    )
    assert len(rewards) == 2


# ---------------------------------------------------------------------------
# Configuration overrides
# ---------------------------------------------------------------------------


def test_custom_starting_gold():
    before_each(configuration={"startingGold": 500})
    state = env.reset()
    gold = state[0]["observation"]["gold"]
    assert gold[0] == 500
    assert gold[1] == 500


def test_custom_episode_steps():
    before_each(configuration={"episodeSteps": 5})
    assert env.configuration.episodeSteps == 5

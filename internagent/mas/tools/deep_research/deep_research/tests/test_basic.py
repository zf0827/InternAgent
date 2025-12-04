import os
import types

def test_package_import():
    import deep_research
    assert hasattr(deep_research, 'AutogenDeepSearchAgent')

def test_agent_instantiation():
    from deep_research import AutogenDeepSearchAgent
    agent = AutogenDeepSearchAgent()
    assert isinstance(agent, AutogenDeepSearchAgent)


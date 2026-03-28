from orchestrator.tools.artifact_tool import ARTIFACT_TOOLS
from orchestrator.tools.form_stubs import FORM_TOOLS
from orchestrator.tools.sample_agents import SAMPLE_AGENT_TOOLS
from orchestrator.tools.stubs import TOOLS as _STUB_TOOLS

DEFAULT_TOOLS = [*_STUB_TOOLS, *SAMPLE_AGENT_TOOLS, *FORM_TOOLS, *ARTIFACT_TOOLS]
TOOLS = DEFAULT_TOOLS

__all__ = ["DEFAULT_TOOLS", "TOOLS"]

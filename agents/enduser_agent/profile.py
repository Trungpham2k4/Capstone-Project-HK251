# -------------------------
# Profile module
# -------------------------

from agents.base_agent.profile import ProfileModule

class EndUserProfile(ProfileModule):

    def system_prompt(self) -> str:
        """Return the system prompt block representing profile."""
        return "Enduser Agent specialized in interacting with end users."
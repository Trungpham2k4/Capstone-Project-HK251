# -------------------------
# Profile module
# -------------------------

from agents.base_agent.profile import ProfileModule

class InterviewerProfile(ProfileModule):

    def system_prompt(self) -> str:
        """Return the system prompt block representing profile."""
        return "You are an interviewer agent."
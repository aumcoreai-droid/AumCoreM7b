# Role Mode Switcher – Phase-1 Rule-based
class RoleModeSwitcher:
    def __init__(self):
        self.current_role = "default"

    def switch_role(self, role: str) -> str:
        valid_roles = ["default", "assistant", "admin", "observer"]
        if role in valid_roles:
            self.current_role = role
            return f"Switched to role: {role}"
        else:
            return f"Invalid role: {role}, remaining in {self.current_role}"

# Test
if __name__ == "__main__":
    rms = RoleModeSwitcher()
    print(rms.switch_role("assistant"))
    print(rms.switch_role("admin"))
    print(rms.switch_role("invalid_role"))

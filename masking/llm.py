from masking.mask import mask, unmask


class MaskingLLM:
    def __init__(self, llm, identifiers):
        self.llm = llm
        self.identifiers = identifiers

    def generate(self, system_prompt, user_prompt):
        # system prompt usually has no patient data. Masking it costs nothing and is safe if PHI ever slips in.
        masked_system, map1 = mask(system_prompt, self.identifiers)
        masked_user, map2 = mask(user_prompt, self.identifiers)

        answer = self.llm.generate(masked_system, masked_user)
        combined = {**map1, **map2}

        # unmask the answer and return it
        return unmask(answer, combined)

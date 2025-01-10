class Direct:
    def approximate(self, g):
        g_hat = g

        return g_hat

    def reconstruct(self, g_hat_attack, uncertainty_hat):
        g_attack = g_hat_attack
        uncertainty = uncertainty_hat

        return g_attack, uncertainty

class Subgraph:
    def approximate(self, g):
        raise NotImplementedError

    def reconstruct(self, g_hat_attack, uncertainty_hat):
        raise NotImplementedError

import torch
import dgl
import sklearn.decomposition as skl

class Direct:
    def approximate(self, g):
        return g

    def reconstruct(self, g_hat_attack, uncertainty_hat):
        return g_hat_attack, uncertainty_hat

class Sequential:
    def __init__(self, *proxies):
        self.proxies = proxies

    def approximate(self, g):
        for proxy in self.proxies:
            g = proxy.approximate(g)

        return g

    def reconstruct(self, *args):
        for proxy in reversed(self.proxies):
            args = proxy.reconstruct(*args)

        return args

class Subgraph:
    def subgraph(self, g):
        raise NotImplementedError

    def approximate(self, g):
        g_hat = self.subgraph(g)

        self.original = g
        self.mapping = g_hat.ndata[dgl.NID]

        return g_hat

    def reconstruct(self, g_hat_attack, uncertainty_hat):
        g_attack = self.original
        mapping = self.mapping

        m = len(mapping)
        n = g_attack.num_nodes()
        o = g_hat_attack.num_nodes() - m

        g_attack.add_nodes(o)

        for field in ['feature', 'label', 'sensitive']:
            g_attack.ndata[field][n:] = g_hat_attack.ndata[field][m:]

        u, v = g_hat_attack.edges()

        new_u = torch.where(torch.logical_and(u >= m, v < m))
        new_v = torch.where(torch.logical_and(v >= m, u < m))
        new_uv = torch.where(torch.logical_and(u >= m, v >= m))

        offset = n - m
        g_attack.add_edges(u[new_u] + offset, mapping[v[new_u]])
        g_attack.add_edges(mapping[u[new_v]], v[new_v] + offset)
        g_attack.add_edges(u[new_uv] + offset, v[new_uv] + offset)

        uncertainty = torch.zeros(g_attack.num_nodes(), device=g_attack.device)
        uncertainty[mapping] = uncertainty_hat

        return g_attack, uncertainty

class KHops(Subgraph):
    def __init__(self, k, root=None):
        self.k = k
        self.root = root

    def subgraph(self, g):
        if self.root is None:
            degree = g.in_degrees() + g.out_degrees()
            self.root = torch.argmax(degree)

        g_hat, _ = dgl.khop_in_subgraph(g, self.root, self.k)

        return g_hat

class PCA:
    def __init__(self, components):
        self.features = None
        self.mean = None
        self.pca = skl.PCA(components)

    def approximate(self, g):
        self.features = g.ndata['feature']
        g.ndata['feature'] = torch.tensor(
                self.pca.fit_transform(self.features.cpu()),
                dtype=self.features.dtype,
                device=self.features.device,
        )

        return g

    def reconstruct(self, g_attack, uncertainty):
        g_attack.ndata['feature'] = self.pca.inverse_transform(g_attack.ndata['feature'].cpu()).to(self.features.device).type(self.features.dtype)
        g_attack.ndata['feature'][:len(self.features)] = self.features

        return g_attack, uncertainty

"""Non-monotone stabilization state (paper section 2.4, rules NmD/15)."""


class NMSState:
    def __init__(self, merit0, opts):
        self.opts = opts
        self.checkpoint_merits = [merit0]
        self.reference = merit0
        self.delta = opts.delta0
        self.checkpoint_k = 0

    def d_allowed(self, k):
        return k < self.checkpoint_k + self.opts.n_bar

    def after_d(self):
        self.delta *= self.opts.beta

    def new_checkpoint(self, merit, k):
        self.checkpoint_merits.append(merit)
        recent = self.checkpoint_merits[-max(self.opts.m_bar, 1):]
        self.reference = max(recent)          # rule (15)
        self.delta = self.opts.delta0
        self.checkpoint_k = k + 1

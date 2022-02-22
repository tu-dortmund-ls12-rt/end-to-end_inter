"""Augmented job chains.

Source:
https://github.com/tu-dortmund-ls12-rt/end-to-end"""


class AugJobChain:
    """Augmented job chain."""

    def __init__(self, job_chain=[], ext_activity=None, actuation=None):
        """Create an augmented job chain."""
        self.job_chain = job_chain  # list of jobs
        self.ext_activity = ext_activity  # external activity
        self.actuation = actuation  # actuation

    def add_job(self, job):
        """Add a job to the job chain."""
        self.job_chain.append(job)

    def set_ext_activity(self, value):
        """Set external activity of the augmented job chain."""
        self.ext_activity = value

    def set_actuation(self, value):
        """Set actuation of the augmented job chain."""
        self.actuation = value

    def length(self):
        """Return the length of the augmented job chain."""
        return (self.actuation-self.ext_activity)

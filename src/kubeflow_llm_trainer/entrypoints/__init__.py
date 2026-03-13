"""Container entrypoints for LLM backends.

These modules are invoked inside the training Pods by the Kubeflow Trainer
controller.  They read configuration from environment variables set by the
SDK and delegate to the actual training framework CLI.
"""

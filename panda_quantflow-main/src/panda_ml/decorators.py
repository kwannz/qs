"""
Node decorators for PandaAI ML
"""

def node_spec(name, description, inputs, outputs, params=None):
    """Node specification decorator.
    
    Args:
        name (str): Node name
        description (str): Node description
        inputs (list): List of input names
        outputs (list): List of output names
        params (dict, optional): Node parameters. Defaults to None.
    """
    def decorator(cls):
        cls._node_name = name
        cls._node_description = description
        cls._node_inputs = inputs
        cls._node_outputs = outputs
        cls._node_params = params or {}
        return cls
    return decorator 
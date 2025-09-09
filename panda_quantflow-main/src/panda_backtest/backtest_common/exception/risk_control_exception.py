class RiskControlException(Exception):
    def __init__(self, message, code, service):
        super().__init__(message, code)
        self.message = message
        self.code = code
        self.service = service

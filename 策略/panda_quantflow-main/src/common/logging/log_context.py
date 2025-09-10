import contextvars
import logging

# Context variables
request_id_var = contextvars.ContextVar("req_id", default="-")
workflow_run_id_var = contextvars.ContextVar("workflow_run_id", default="-")
backtest_id_var = contextvars.ContextVar("backtest_id", default="-")


def set_request_id(req_id: str):
    return request_id_var.set(req_id)


def reset_request_id(token):
    try:
        request_id_var.reset(token)
    except Exception:
        pass


def set_workflow_run_id(wf_id: str):
    return workflow_run_id_var.set(wf_id)


def reset_workflow_run_id(token):
    try:
        workflow_run_id_var.reset(token)
    except Exception:
        pass


def set_backtest_id(bt_id: str):
    return backtest_id_var.set(bt_id)


def reset_backtest_id(token):
    try:
        backtest_id_var.reset(token)
    except Exception:
        pass


class ContextFilter(logging.Filter):
    """Injects contextvars into LogRecord for formatting"""

    def filter(self, record: logging.LogRecord) -> bool:
        # Attach or default
        setattr(record, 'req_id', request_id_var.get())
        setattr(record, 'workflow_run_id', workflow_run_id_var.get())
        setattr(record, 'backtest_id', backtest_id_var.get())
        return True


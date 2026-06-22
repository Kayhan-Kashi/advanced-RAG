from typing import Type, Dict, Any, Optional
from injector import Injector, inject
import inspect


class MessageBus:
    @inject
    def __init__(self, injector: Injector):
        self._handlers: Dict[Type, Type] = {}
        self._injector = injector

    def register(self, command_type: Type, handler_type: Type):
        self._handlers[command_type] = handler_type

    async def dispatch(self, command: Any, db: Optional[Any] = None):
        """Dispatch command/event to the registered handler"""

        handler_cls = self._handlers.get(type(command))
        if handler_cls is None:
            raise ValueError(f"No handler registered for {type(command)}")

        handler = self._injector.get(handler_cls)

        sig = inspect.signature(handler.handle)

        # call handler without awaiting yet
        if "db" in sig.parameters:
            result = handler.handle(command, db=db)
        else:
            result = handler.handle(command)

        # ✅ async generator (streaming handler)
        if inspect.isasyncgen(result):
            return result

        # ✅ coroutine
        if inspect.isawaitable(result):
            return await result

        # ✅ direct return value
        return result
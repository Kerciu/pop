from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstrakcyjna klasa bazowa dla wszystkich agentów.
    Gwarantuje, że każdy agent będzie pasował do pętli treningowej.
    """

    @abstractmethod
    def get_action(self, state, epsilon=0.0) -> int:
        """
        Przyjmuje stan (Tensor lub obiekt) i zwraca ID akcji (int).
        Epsilon służy do sterowania eksploracją (dla RL).
        """
        pass

    def update(self, state, action, reward, next_state, done):
        """
        Metoda do uczenia się (opcjonalna, np. bot losowy jej nie potrzebuje).
        """
        pass

    def save(self, path):
        """Zapisuje model na dysku."""
        pass

    def load(self, path):
        """Wczytuje model z dysku."""
        pass

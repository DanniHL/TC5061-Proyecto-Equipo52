import pandas as pd
import pytest


@pytest.fixture
def viajes_p1_ejemplo():
    """Genera un conjunto pequeño de viajes para las pruebas de P1."""
    return pd.DataFrame(
        {
            "Inicio_del_viaje": pd.to_datetime(
                [
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-02 10:00:00",
                    "2026-01-01 11:00:00",
                ]
            ),
            "Origen_Id": [10, 10, 10, 20],
        }
    )

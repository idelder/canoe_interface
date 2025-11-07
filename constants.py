"""
Some Enums to avoid hard coding strings elsewhere.
For readability, maintainability, and to reduce typos.
"""

#########################################
#   Enumerators for CANOE data sets
#########################################

from enum import Enum
from typing import Set, List

class SortableEnum(Enum):
    """
    An Enum that supports less-than comparisons based on value.
    """
    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.value < other.value

class Region(SortableEnum):
    """
    Viable regions in CANOE data sets.
    """
    ALBERTA = 'AB'
    BRITISH_COLUMBIA = 'BC'
    MANITOBA = 'MB'
    NEW_BRUNSWICK = 'NB'
    NEWFOUNDLAND_AND_LABRADOR = 'NLLAB'
    NOVA_SCOTIA = 'NS'
    ONTARIO = 'ON'
    PRINCE_EDWARD_ISLAND = 'PEI'
    QUEBEC = 'QC'
    SASKATCHEWAN = 'SK'
    USA = 'USA'
    NONE = ''

class Sector(SortableEnum):
    """
    Viable sectors in CANOE data sets.
    """
    AGRICULTURE = 'AGRI'
    COMMERCIAL = 'COM'
    ELECTRICITY = 'ELC'
    FUEL = 'FUEL'
    INDUSTRY = 'IND'
    RESIDENTIAL = 'RES'
    TRANSPORTATION = 'TRP'

class Variant(SortableEnum):
    """
    Viable sector variants in CANOE data sets.
    """
    HIGH_RESOLUTION = 'HR'
    CURRENT_MEASURES = 'CM'
    NET_ZERO = 'NZ'

class Feature(SortableEnum):
    """
    Viable additional features in CANOE data sets.
    """
    DEMAND = 'DEM'
    ENDOGENOUS_INTERTIE = 'EINT'
    BOUNDARY_INTERTIE = 'BINT'
    NONE = ''

class Level(SortableEnum):
    """
    Sector resolution levels for UI
    """
    HIGH_RESOLUTION = 'HIGH'
    LOW_RESOLUTION = 'LOW'
    EXCLUDED = '-'

#########################################
#   UI constants
#########################################

CONFIG_FILE: str = "input/saved_config.json"
"""File location for saved configuration of UI options"""

LOW_SCENARIOS: List[str] = [
    "Current Measures",
    "Global Net Zero"
]
"""UI naming for low resolution scenarios"""

DEFAULT_LOW: str = 'Current Measures'
"""Default low resolution scenario (UI name)"""

TABLE_SECTORS: List[Sector] = sorted(
    s.value for s in [
        Sector.COMMERCIAL,
        Sector.ELECTRICITY,
        Sector.INDUSTRY,
        Sector.RESIDENTIAL,
        Sector.TRANSPORTATION
    ]
)
"""Sectors that are included in UI options matrix"""

TABLE_REGIONS: List[Region] = sorted(
    r.value for r in [
        Region.ALBERTA,
        Region.BRITISH_COLUMBIA,
        Region.MANITOBA,
        Region.NEW_BRUNSWICK,
        Region.NEWFOUNDLAND_AND_LABRADOR,
        Region.NOVA_SCOTIA,
        Region.ONTARIO,
        Region.PRINCE_EDWARD_ISLAND,
        Region.QUEBEC,
        Region.SASKATCHEWAN
    ]
)
"""Regions that are included in UI options matrix"""

#########################################
#   Database processing constants
#########################################

SCHEMA_FILE: str = "input/schema.sql"
"""File location for CANOE database schema SQL"""

INDEX_TABLES: Set[str] = {
    'CommodityType','Operator','TechnologyType','TimePeriodType',
    'DataQualityCredibility','DataQualityGeography',
    'DataQualityStructure','DataQualityTechnology','DataQualityTime',
    'TechnologyLabel','CommodityLabel','DataSourceLabel'
}
"""SQLite tables that are handled by the schema and need not be transferred"""

LTT_DEFAULT: int = 40
"""Default lifetime for technologies in Temoa if not specified elsewhere"""
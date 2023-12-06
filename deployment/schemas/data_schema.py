from pydantic import BaseModel
from typing import Optional, List


class SingleDataScheme(BaseModel):
    #goal is to list datatypes of all required fields (the selected features) and also for the non-required fields.
    #if you use e.g. ClientID: Optional[str] then the field is REQUIRED but the value can be missing (None) or str.
    #Hence do not use Optional for non-required fields as an error will be raised when field is missing.

    #all fields from the raw input data (selected features non optional)
    ClientID: str = None #If ClientID not specified it will be given None default value
    Astatus: Optional[str] #REQUIRED
    Duration: Optional[float] #REQUIRED
    CHistory: Optional[str] #REQUIRED
    Purpose: Optional[str] #REQUIRED
    Camount: float = None
    Saccount: str = None
    Etime: str = None
    IRate: int = None
    Pstatus: str = None
    Debtors: str = None 
    Residence: int = None
    Property: str = None
    Age: float = None
    Iplans: str = None
    Housing: Optional[str] #REQUIRED
    Ncredits: int = None
    Job: str = None
    Depend: int = None
    Phone: str = None
    Fworker: str = None
    Status: int = None


class RawDataScheme(BaseModel):
    inputs: List[SingleDataScheme]

    #default input values (example showcase the API, equals the first test data point)
    class Config:
        json_schema_extra = {
            "example": {
                "inputs": [
                    {
                        'ClientID': "random_client",
                        'Astatus': 'A11',
                        'Duration': 36,
                        'CHistory': 'A33',
                        'Purpose': 'A46',
                        'Camount': 6887,
                        'Saccount': 'A61',
                        'Etime': 'A73',
                        'IRate': 4,
                        'Pstatus': 'A93',
                        'Debtors': 'A101',
                        'Residence': 3,
                        'Property': 'A122',
                        'Age': 29.0,
                        'Iplans': 'A142',
                        'Housing': 'A152',
                        'Ncredits': 1,
                        'Job': 'A173',
                        'Depend': 1,
                        'Phone': 'A192',
                        'Fworker': 'A201',
                        'Status': 1
                    }
                ]
            }
        }
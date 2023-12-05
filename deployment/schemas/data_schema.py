from pydantic import BaseModel
from typing import Optional, List


class SingleDataScheme(BaseModel):
    #all fields from the raw input data (selected features non optional)
    ClientID: Optional[str]
    Astatus: str
    Duration: float
    CHistory: str
    Purpose: str
    Camount: Optional[float]
    Saccount: Optional[str]
    Etime: Optional[str]
    IRate: Optional[int]
    Pstatus: Optional[str]
    Debtors: Optional[str]
    Residence: Optional[int]
    Property: Optional[str]
    Age: Optional[float]
    Iplans: Optional[str]
    Housing: str
    Ncredits: Optional[int]
    Job: Optional[str]
    Depend: Optional[int]
    Phone: Optional[str]
    Fworker: Optional[str]
    Status: Optional[int]


class RawDataScheme(BaseModel):
    inputs: List[SingleDataScheme]

    #default input values (example showcase the API, equals the first test data point)
    class Config:
        schema_extra = {
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
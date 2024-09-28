from .record_type import RecordType
from .dsspath import DssPath
from datetime import datetime

class Catalog:
    """manage list of objects inside a DSS database"""
    def __init__(self, items, recordTypes):
        self.rawCatalog = items
        self.rawRecordTypes = recordTypes
        self.timeSeriesDictNoDates = {}  # key is path without date, value is a list dates
        self.recordTypeDict = {} # key is path w/o date, value is recordType
        self.__create_condensed_catalog()
 
    def __create_condensed_catalog(self):
        """
        Condensed catalog combines time-series records into a single condensed path.
        Other record types are not condensed.
        Time-series records must match all parts except the D (date) part to be combined.
        """
        self.items = []
        for i in range(len(self.rawCatalog)):
            rawPath = self.rawCatalog[i]
            recordType = RecordType.RecordTypeFromInt(self.rawRecordTypes[i])
            path = DssPath(rawPath, recordType)
    
            # Debugging: Print the rawPath being processed
            print(f"Processing rawPath: {rawPath}")
    
            # Check for 'TS-PATTERN' in the path itself
            if 'TS-PATTERN' in rawPath or 'TS-PATTERN' in path.D:
                print(f"Skipping invalid path: {rawPath}")
                continue  # Skip this entry
    
            # If timeseries - accumulate dates within a dataset
            if path.is_time_series():
                cleanPath = str(path.path_without_date())
                self.recordTypeDict[cleanPath] = recordType
                tsRecords = self.timeSeriesDictNoDates.setdefault(cleanPath, [])
                print(f"Processing time series path: {path.D}")
    
                # Attempt to parse the date only if path.D is valid
                try:
                    t = datetime.strptime(path.D, "%d%b%Y")  # Adjust format if necessary
                    tsRecords.append(t)
                except ValueError:
                    print(f"Date parsing error for path: {path.D}")
                    continue  # Skip if there's a parsing error
    
            elif recordType in [RecordType.PairedData, RecordType.Grid, RecordType.Text,
                                RecordType.LocationInfo, RecordType.Array]:
                self.recordTypeDict[str(path)] = recordType
                self.items.append(path)
            else:
                raise Exception(f"Unsupported record_type: {recordType}")

        # Process time series dates
        for key in self.timeSeriesDictNoDates:
            dateList = sorted(self.timeSeriesDictNoDates[key])
            condensedDpart = dateList[0].strftime("%d%b%Y")
            if len(dateList) > 1:
                condensedDpart += "-" + dateList[-1].strftime("%d%b%Y")
            # Insert condensed D part into path used as key
            rt = self.recordTypeDict[key]
            p = DssPath(key, rt)
            p.D = condensedDpart
            self.items.append(p)


    def print(self):
        for ds in self.items:
            print(ds)

    def __iter__(self):
        self.index = 0  # Initialize the index to 0
        return self

    def __next__(self):
        if self.index < len(self.items):
            result = self.items[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

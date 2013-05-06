Patent Example
=================

This example shows `dedupe` being used to disambiguate data on
inventors from the PATSTAT international patent data file.

The example illustrates a few more elaborate features of PATSTAT:
1. Set and class comparisons, useful for records where sets are a
record attribute (e.g., for voters, "candidates donated to")
2. Geographic distance comparison, via a Haversine distance
3. Set and geographic distance blocking predicates

The example also illustrates a potential two-stage disambiguation
strategy that targets two separate groups of inventors. The first
round goes after the entire group of inventors, with precision-recall
critera favoring more precise disambiguation. The second round selects
only the most prolific innovators, and further disambiguates with
precision-recall criteria favoring recall. The reflects the fact that
prolific innovators may have lots of name variance that should be
ignored, while rare innovators (especially individuals) are
distinguished by minor differences. 

For instance, "IBM", "International Business Machines", and
"Internat. Bus. Machines" are all the same company; the dedupe
algorithm should accept this high level of name variance. But doing so
with the same dataset containing "John R. Smith", "John H. Smith", and
"John A. Smith" would aggregate together distinct individuals. By
separating the disambiguation into stages, we enable the algorithm to
learn different settings for these two cases. 

Data 
-----------

- The `patstat_input.csv` file contains data on Dutch
  innovators. Fields are:
  - Person: the numerical identifier assigned by the Dutch patent
  office. This is not guaranteed to be a unique identifier for all
  instances of a single inventor
  - Name: the inventor name
  - Coauthors: coauthors listed on patents assigned to this inventor
  - Class: the 4-digit IPC technical codes listed on patents assigned
  to this inventor
  - Lat, Lng: the latitude and longitude, geocoded from the inventor's address. 0.0
  indicates no geocode-able address was found
- The `patstat_reference.csv` file contains reference data provided by
  KU Leuven, mapping the Person data to a manually-checked unique
  identifier. Fields are:
  - `person_id`: the PATSTAT unique identifier, equivalent to Person above
  - `leuven_id`: the hand-checked unique identifier; there is a 1:many
  leuven_id:person_id relationship
  - `person_name`: the raw person name matching this person_id

Running the example
-------------------

```python

# To run the disambiguation itself:

python -u patent_example.py 

# To check the precision-recall relative to the provided reference
# data:

python compute_precision_recall patstat_output_2.csv patstat_reference.csv

```


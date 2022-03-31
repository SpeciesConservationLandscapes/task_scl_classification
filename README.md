Species Conservation Landscapes Classification task
-----------------------

Task for classifying SCL polygons produced by task_scl_eff_pot_hab into species/restoration/survey

## Usage

*All parameters may be specified in the environment as well as the command line.*

```
/app # python task.py --help
usage: task.py [-h] [-d TASKDATE] [-s SPECIES] [-u] [--scenario SCENARIO] [--overwrite]

optional arguments:
  -h, --help            show this help message and exit
  -d TASKDATE, --taskdate TASKDATE
  -s SPECIES, --species SPECIES
  -u, --use-cache
  --scenario SCENARIO
  --overwrite           overwrite existing outputs instead of incrementing
```

### License
Copyright (C) 2022 Wildlife Conservation Society
The files in this repository  are part of the task framework for calculating 
Human Impact Index and Species Conservation Landscapes (https://github.com/SpeciesConservationLandscapes) 
and are released under the GPL license:
https://www.gnu.org/licenses/#GPL
See [LICENSE](./LICENSE) for details.

# Spectra

Spectra are stored in a spectra SQLite database or in a PostgreSQL database.
Spectra can be searched by category, subcategory, name, type and phase.
Load dependent spectra can be interpolated or extrapolated to different loads.

OpenDSS requires assigning spectra to loads, PV Systems and other
Power Conversion Elements (PCElements) for harmonic analysis.
If no spectrum is provided, default spectra are used.

## Spectra Types

in OpenDSS spectra are not necessarily current spectra but may also be voltage spectra.
In the documentation the following information can be found:

| PCElement  | Spectrum Type     | default spectrum |
|------------|-------------------|------------------|
| Load       | current           | defaultload      |
| Generator  | current / voltage | default          |
| IndMach012 | current / voltage | ?                |
| Storage    | current /voltage  | default          |
| VSource    | voltage           | defaultvsource   |
| ISource    | current           | default          |
| GICLine    | ?                 | defaultvsource   |

- For Generator: Voltage behind Xd" for machine - default. Current injection for inverter.
- For Induction Machine:  Voltage behind Xd", or blocked rotor, for machine - default.
- For Storage:  Current injection is assumed for inverter.

## OpenDSS Default Spectra

| Name           | Harmonics                           | Magnitude                                                 | Angle                                              |
|----------------|-------------------------------------|-----------------------------------------------------------|----------------------------------------------------|
| default        | '1 3 5 7 9 11 13'                   | '100 33 20 14 11 9 7'                                     | '0 0 0 0 0 0 0'                                    |
| defaultload    | '1 3 5 7 9 11 13'                   | '100 1.5 20 14 1 9 7'                                     | '0 180 180 180 180 180 180'                        |
| defaultvsource | '1'                                 | '100'                                                     | '0'                                                |
| linear         | '1'                                 | '100'                                                     | '0'                                                |
| pwm6           | '1 3 5 7 9 11 13 15 17 19 21 23 25' | '100 4.4 76.5 62.7 2.9 24.8 12.7 0.5 7.1 8.4 0.9 4.4 3.3' | '-103 -5 28 -180 -33 -59 79 36 -253 -124 3 -30 86' |
| dc6            | '1 3 5 7 9 11 13 15 17 19'          | '100 1.2 33.6 1.6 0.4 8.7 1.2 0.3 4.5 1.3'                | '-75 28 156 29 -91 49 54 148 -57 -46'              |
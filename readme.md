# APLib

APLib is designed for easy integration into an existing object detection project. It provides: 
- A simple function to evaluate mAP
- Utilities to convert between different formats of data.

# Disclaimer

I refactored [Cartucho/mAP](https://github.com/Cartucho/mAP), which was directly adapted from the PASCAL VOC's matlab code. It is great that the original code has been tested against the matlab version, however, its style is too matlab-like. 

# Usage
See the example under `examples` directory. It is validated to provide the same results as in [Cartucho/mAP](https://github.com/Cartucho/mAP).

```sh
$ cd examples; python example.py
```

```python
from __future__ import annotations

import matgl
from matcalc.elasticity import ElasticityCalc
from matgl.ext.ase import PESCalculator
from pymatgen.ext.matproj import MPRester

potential = matgl.load_model("TensorNet-MatPES-PBE-v2025.2-PES")
ase_calc = PESCalculator(potential)
calculator = ElasticityCalc(ase_calc)
```

    /Users/shyue/repos/matgl/src/matgl/apps/pes.py:69: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.element_refs = AtomRef(property_offset=torch.tensor(element_refs, dtype=matgl.float_th))
    /Users/shyue/repos/matgl/src/matgl/apps/pes.py:75: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_mean", torch.tensor(data_mean, dtype=matgl.float_th))
    /Users/shyue/repos/matgl/src/matgl/apps/pes.py:76: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
      self.register_buffer("data_std", torch.tensor(data_std, dtype=matgl.float_th))


Let us obtain the structure of Si from the Materials Project API


```python
mpr = MPRester()
```


```python
si = mpr.get_structure_by_material_id("mp-149")
print(si)
```

    Full Formula (Si2)
    Reduced Formula: Si
    abc   :   3.849278   3.849279   3.849278
    angles:  60.000012  60.000003  60.000011
    pbc   :       True       True       True
    Sites (2)
      #  SP        a      b      c    magmom
    ---  ----  -----  -----  -----  --------
      0  Si    0.875  0.875  0.875        -0
      1  Si    0.125  0.125  0.125        -0



```python
pred = calculator.calc(si)
```

For comparison, let's obtain the DFT computed values from Materials Project


```python
mp_data = mpr.get_summary_by_material_id("mp-149")
```


```python
print(
    f"K_VRH: TensorNet-MatPES-PBE = {pred['bulk_modulus_vrh']}; DFT = {mp_data['bulk_modulus']['vrh']}"
)
print(
    f"G_VRH: TensorNet-MatPES-PBE = {pred['shear_modulus_vrh']}; DFT = {mp_data['shear_modulus']['vrh']}"
)
```

    K_VRH: TensorNet-MatPES-PBE = 101.15424648468968; DFT = 88.916
    G_VRH: TensorNet-MatPES-PBE = 62.546024424713266; DFT = 62.445

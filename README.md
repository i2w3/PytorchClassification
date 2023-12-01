# PytorchClassification
Use Pytorch to classify images

# Experimental Result
<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Transforms</th>
    <th colspan="3">Experimental</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">ResNet18</td>
    <td>baseline</td>
    <td>0.8333</td>
    <td>0.8558</td>
    <td>0.8446</td>
  </tr>
  <tr>
    <td>cutmix or mixup</td>
    <td>0.8782</td>
    <td>0.8494</td>
    <td>0.8878</td>
  </tr>
  <tr>
    <td rowspan="2">SE-ResNet18</td>
    <td>baseline</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>cutmix or mixup</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">ResNet50</td>
    <td>baseline</td>
    <td>0.8510</td>
    <td>0.8638</td>
    <td>0.8590</td>
  </tr>
  <tr>
    <td>cutmix or mixup</td>
    <td>0.8750</td>
    <td>0.8830</td>
    <td>0.8814</td>
  </tr>
  <tr>
    <td rowspan="2">SE-ResNet50</td>
    <td>baseline</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>cutmix or mixup</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">ViT-b-16</td>
    <td>baseline</td>
    <td>0.8253</td>
    <td>0.8109</td>
    <td>0.8397</td>
  </tr>
  <tr>
    <td>cutmix or mixup</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
</table>
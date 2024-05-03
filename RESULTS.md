# Results with newer Pytorch3D version

We only provide these numbers for reference to check the setup with more recent Pytorch3D versions.
Please cite the results reported in the published paper in scientific works.

## Trajecotry fitting sphere

Updated numbers for Table 1 from the paper:

<table>
<tr>
    <th colspan="2">&nbsp;</th>
    <th colspan="3">resulting radius error</th>
</tr>
<tr>
    <th>scenario</th>
    <th>variant</th>
    <th>min</th>
    <th>mean</th>
    <th>max</th>
</tr>
<tr>
    <td rowspan="2">w/ gravity</td>
    <td>w/o toc</td>
    <td>9e-5</td>
    <td>0.038</td>
    <td>0.219</td>
</tr>
<tr>
    <td>w/ toc</td>
    <td>2e-6</td>
    <td>0.006</td>
    <td>0.047</td>
</tr>
<tr>
    <td rowspan="2">w/o gravity</td>
    <td>w/o toc</td>
    <td>fails</td>
    <td>fails</td>
    <td>fails</td>
</tr>
<tr>
    <td>w/ toc</td>
    <td>2e-4</td>
    <td>0.002</td>
    <td>0.006</td>
</tr>
</table>

## Fitting to depth measurements

Updated numbers from Table 2 from the paper:

<table>
<tr><th>&nbsp;</th><th colspan="2">sphere</th><th>cube</th></tr>
<tr><th>error</th><th>w/o gravity</th><th>w/ gravity</th><th>w/o gravity</th></tr>
<tr><td>init pos </td><td> 0.040 </td><td> 0.040 </td><td> 0.040 </td></tr>
<tr><td>pos frame fit </td><td> 0.056 </td><td> 0.056 </td><td> 0.077 </td></tr>
<tr><td>pos traj. fit </td><td> 0.031 </td><td> 0.039 </td><td> 0.022 </td></tr>
<tr><td>init rot </td><td> 0.135 </td><td> 0.135 </td><td> 0.135 </td></tr>
<tr><td>rot frame fit </td><td> 0.135 </td><td> 0.135 </td><td> 0.001 </td></tr>
<tr><td>rot traj. fit </td><td> 0.135 </td><td> 0.134 </td><td> 0.007 </td></tr>
<tr><td>init size </td><td> 0.512 </td><td> 0.512 </td><td> 0.512 </td></tr>
<tr><td>size frame fit </td><td> 0.163 </td><td> 0.163 </td><td> 0.137 </td></tr>
<tr><td>size traj. fit </td><td> 0.022 </td><td> 0.024 </td><td> 0.034 </td></tr>
</table>

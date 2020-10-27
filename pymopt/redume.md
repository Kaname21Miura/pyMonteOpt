
# PyMonteOpt
PyMonteOpt is Monte Carlo simulation modules for optics and biology.
### Classes
* [SolidPlateModel](#solidPlateModel)
* [VoxelPlateModel](#VoxelDicomModel)
* [VoxelDicomModel](#VoxelDicomModel)
* [Fuluence](#Fuluence)



## SolidPlateModel<a id="solidPlateModel"></a>


## VoxelPlateModel<a id="VoxelPlateModel"></a>
VoxelPlateModel is voxel-baced multilayer Monte Carlo model.


## VoxelDicomModel<a id="VoxelDicomModel"></a>
```python
class pymopt.voxel.VoxelDicomModel(
  nPh = 1000, model_type = 'binary',d_beam = 0
  fluence_mode = False, nr = 50, nz = 20, dr = 0.1, dz = 0.1,
  dtype = 'float32'
  )
```
VoxelDicomModel class は、Dicom 画像を用いたモンテカルロ法を計算するモジュールです。メソッドで Dicom 画像の簡単な編集が可能です。

### Parameters:
* ###### nPh: int, default = 1000</br>
  光子数を設定します。
* ###### model_type: {'binary', 'liner'}, default = 'binary'</br>
  Dicom画像の表現方法を決定します。'binary'は画像の空隙部を0それ以外を１で表現します。'liner'は画像のグレースケールによって散乱係数を変化させるモデルです。散乱係数と画像の輝度値(br)の関係は、16Bit画像の場合以下の式のようになります。
```math
µs = 1.67e-3*Br - 10.931
```
* ###### beam_w: float, default = 0</br>
* ###### fluence_mode: {'2D', '3D'} or False, default = False</br>
* ###### nr: float, default = 0</br>
* ###### nz: float, default = 0</br>
* ###### dr: float, default = 0</br>
* ###### dz: float, default = 0</br>
* ###### dtype: {'float32', 'float64'}, default = 'float32'</br>

##### note
### Example
詳しい例はチュートリアルを参照してください。
```python
from pymopt.voxel import VoxelDicomModel
dicom_path = '../DicomData'
monte_params = {
 'nPh':1e7,'model_type':'binary'
}

params = {
  'ct_th':0.3, 'skin_th':3.,
  'tr_ma':0.5, 'tr_ms':12.,
}


dcm = VoxelDicomModel()
dcm.import_dicom(dicom_path)
dcm.set_monte_params(monte_params)
dcm.set_params(params)
dcm.build()

dcm.start()
dcm.get_result()
```

### Methods
* ###### Basic methods
Method name|contents
:---|:---
[build](#solidPlateModel)()| 計算するための３Dモデルを生成する
[start](#set_monte_params)()|
[save_result](#save_result)(path, coment)|結果をファイルに保存する
[set_model_params](#set_params)(params)| モデルのパラメーターを設定する
[set_monte_params](#set_monte_params)(**params)|
[get_result](#get_result)()|
[get_fluence](#get_fluence)()|
[get_model_params](#get_model_params)()| モデルのパラメーターを取得する

* ###### uniqe methods
Method name|contents
:---|:---
[import_dicom](#import_dicom)(path)|
[display_cross_section](#display_cross_section)()|
[rot180_y](#rot180_y)()|
[rot90_z](#rot90_z)()|
[trim_area](#trim_area)()|
[set_trim_pixel](#set_trim_pixel)()|
[check_threshold](#check_threshold)()|
[set_threshold](#set_threshold)()|
[set_trim](#set_trim)()|
[display_histmap](#display_histmap)()|
[reset_setting](#reset_setting)()|

#### build()<a id="build"></a>
モデルクラスに計算モデルをビルドします。この時、クラス内の属性に保存されたvoxel画像データは、消去されます。

#### start()<a id="start"></a>
モンテカルロの計算を開始します。

#### save_result(path, coment)<a id="save_result"></a>
計算結果を保存します。
###### Parameters
* path: str </br>
結果を保存するファイル名を入れます。結果は、試料表面から出てきた光子のデータ、モンテカルロの計算条件、Fluenceの結果が保存されます。表面光子データ並びにFluenceの結果は、".pkl.bz2"形式で、計算条件は、".jason"で保存されます。fluence_mode = False の場合は、Fluenceの結果ファイルは生成されません。

* coment: str utf8 </br>
ここに記述されたコメントは、計算条件と共に".jason"で保存されます。

#### set_model_params(params)<a id="build"></a>
モデルの光学パラメーターをセットします。
###### Parameters</br>
Params|Input|contents
:---|:---|:---
th_skin| float, default = 2.|Thickness of skin
th_ct| float, default = 0.03|Thickness of cortical bone
n_sp| float, default = 1.|Refractive index of trabecular space
n_tr| float, default = 1.37|Refractive index of trabecular
n_ct| float, default = 1.37|Refractive index of cortical bone
n_skin| float, default = 1.37|Refractive index of the skin
n_air| float, default = 1.|Refractive index of the atmosphere
ma_sp| float, default = 1e-8|Absorbance coefficient of trabecular space
ma_tr| float, default = 0.011|Absorbance coefficient of trabecular
ma_ct| float, default = 0.011|Absorbance coefficient of cortical bone
ma_skin| float, default = 0.037|Absorbance coefficient of skin
ms_sp| float, default = 1e-8|Scattering coefficient of trabecular space
ms_tr| float, default = 19.1|Scattering coefficient of trabecular
ms_ct| float, default = 19.1|Scattering coefficient of cortical bone
ms_skin| float, default = 18.8|Scattering coefficient of skin
g_sp| float, default = 0.90|Anisotropy coefficient of trabecular space
g_tr| float, default = 0.93|Anisotropy coefficient of trabecular
g_ct| float, default = 0.93|Anisotropy coefficient of cortical bone
g_skin| float, default = 0.93|Anisotropy coefficient of skin

ms_tr は、model_type = 'liner' のとき参照されません。</br>
下からコピーすると楽です。
```python
params = {
'th_skin':2,'th_ct':0.03,
'n_sp':1.,'n_tr':1.37,'n_ct':1.37,'n_skin':1.37,'n_air':1.,
'ma_sp':1e-8, 'ma_tr':0.011,'ma_ct':0.011,'ma_skin':0.037,
'ms_sp':1e-8,'ms_tr':19.1,'ms_ct':19.1,'ms_skin':18.8,
'g_sp':0.90,'g_tr':0.93,'g_ct':0.93,'g_skin':.93,
}
```
#### set_monte_params(**params)<a id="set_monte_params"></a>
Sets the Monte Carlo parameters.
###### Parameters</br>
  Same as init() parameters.

#### get_result()<a id="get_result"></a>
After the calculation by the Monte Carlo method, the position, vector, and weight of the photon that has come out of the sample are obtained.
###### Returns</br>
result: dict
{'p': array(x,y,z), 'v': array(x,y,z), 'w': array(), 'nPh': int}</br>
p is xyz position of phthons, v is vectors, w is photon weights, and nPh is input photon numbers.

#### get_fluence()<a id="get_fluence"></a>
###### Returns</br>

#### get_model_params()<a id="get_model_params"></a>
###### Returns</br>


## References
[1] L. Wang, S. Jacques 1992 <br>
[2]

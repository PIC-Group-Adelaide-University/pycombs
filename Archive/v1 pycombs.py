import numpy as np
import matplotlib.pyplot as plt
import os
import time
import matplotlib.animation as animation
import io

from matplotlib.widgets import Slider, Button, TextBox
from scipy.fft import fft, ifft, fftshift
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator 
from matplotlib.ticker import FixedLocator

t_start = time.time()

DEFAULT_DINT_CSV = '''Wavelength [um],Frequency [THz],Dint/2pi [GHz]
1.6117799427830857,186.0008615582743,0.32
1.611474976888244,186.0360615582743,0.316808
1.6111701263771472,186.0712615582743,0.313632
1.6108653911843245,186.1064615582743,0.310472
1.610560771244355,186.1416615582743,0.307328
1.610256266491867,186.1768615582743,0.3042
1.6099518768615382,186.2120615582743,0.301088
1.6096476022880952,186.24726155827432,0.297992
1.6093434427063151,186.28246155827432,0.294912
1.6090393980510234,186.3176615582743,0.291848
1.608735468257095,186.3528615582743,0.2888
1.6084316532594538,186.3880615582743,0.285768
1.6081279529930737,186.4232615582743,0.282752
1.6078243673929764,186.4584615582743,0.279752
1.607520896394234,186.4936615582743,0.276768
1.6072175399319664,186.52886155827431,0.2738
1.6069142979413438,186.56406155827432,0.270848
1.6066111703575838,186.59926155827432,0.267912
1.606308157115954,186.6344615582743,0.264992
1.6060052581517708,186.6696615582743,0.262088
1.6057024734003984,186.7048615582743,0.2592
1.6053998027972505,186.7400615582743,0.256328
1.6050972462777895,186.7752615582743,0.253472
1.6047948037775266,186.8104615582743,0.250632
1.6044924752320209,186.84566155827432,0.247808
1.6041902605768805,186.88086155827432,0.245
1.603888159747762,186.91606155827432,0.242208
1.6035861726803706,186.9512615582743,0.239432
1.6032842993104595,186.9864615582743,0.236672
1.6029825395738306,187.0216615582743,0.233928
1.602680893406334,187.0568615582743,0.2312
1.602379360743868,187.0920615582743,0.228488
1.6020779415223794,187.1272615582743,0.225792
1.601776635677863,187.16246155827432,0.223112
1.6014754431463618,187.19766155827432,0.220448
1.6011743638639666,187.23286155827432,0.2178
1.6008733977668168,187.2680615582743,0.215168
1.6005725447910992,187.3032615582743,0.212552
1.6002718048730493,187.3384615582743,0.209952
1.5999711779489498,187.3736615582743,0.207368
1.5996706639551317,187.4088615582743,0.2048
1.5993702628279731,187.4440615582743,0.202248
1.5990699745039016,187.47926155827432,0.199712
1.5987697989193905,187.51446155827432,0.197192
1.598469736010962,187.5496615582743,0.194688
1.5981697857151855,187.5848615582743,0.1922
1.5978699479686782,187.6200615582743,0.189728
1.597570222708105,187.6552615582743,0.187272
1.5972706098701779,187.6904615582743,0.184832
1.5969711093916565,187.7256615582743,0.182408
1.5966717212093484,187.7608615582743,0.18
1.5963724452601074,187.79606155827432,0.177608
1.5960732814808356,187.83126155827432,0.175232
1.5957742298084823,187.8664615582743,0.172872
1.5954752901800433,187.9016615582743,0.170528
1.5951764625325628,187.9368615582743,0.1682
1.5948777468031312,187.9720615582743,0.165888
1.5945791429288863,188.0072615582743,0.163592
1.594280650847013,188.0424615582743,0.161312
1.5939822704947435,188.07766155827431,0.159048
1.593684001809356,188.11286155827432,0.1568
1.5933858447281772,188.14806155827432,0.154568
1.5930877991885792,188.1832615582743,0.152352
1.592789865127982,188.2184615582743,0.150152
1.5924920424838516,188.2536615582743,0.147968
1.5921943311937015,188.2888615582743,0.1458
1.5918967311950911,188.3240615582743,0.143648
1.5915992424256273,188.3592615582743,0.141512
1.5913018648229633,188.39446155827432,0.139392
1.591004598324799,188.42966155827432,0.137288
1.59070744286888,188.46486155827432,0.1352
1.5904103983929996,188.5000615582743,0.133128
1.590113464834997,188.5352615582743,0.131072
1.589816642132758,188.5704615582743,0.129032
1.5895199302242147,188.6056615582743,0.127008
1.589223329047345,188.6408615582743,0.125
1.5889268385401738,188.6760615582743,0.123008
1.588630458640772,188.71126155827432,0.121032
1.5883341892872567,188.74646155827432,0.119072
1.5880380304177915,188.78166155827432,0.117128
1.587741981970585,188.8168615582743,0.1152
1.587446043883893,188.8520615582743,0.113288
1.5871502160960171,188.8872615582743,0.111392
1.5868544985453046,188.9224615582743,0.109512
1.5865588911701491,188.9576615582743,0.107648
1.5862633939089894,188.9928615582743,0.1058
1.585968006700311,189.02806155827432,0.103968
1.5856727294826447,189.06326155827432,0.102152
1.5853775621945672,189.0984615582743,0.100352
1.585082504774701,189.1336615582743,0.098568
1.584787557161714,189.1688615582743,0.0968
1.58449271929432,189.2040615582743,0.095048
1.5841979911112787,189.2392615582743,0.093312
1.5839033725513947,189.2744615582743,0.091592
1.5836088635535186,189.3096615582743,0.089888
1.5833144640565462,189.34486155827432,0.0882
1.5830201739994185,189.38006155827432,0.086528
1.5827259933211228,189.4152615582743,0.084872
1.582431921960691,189.4504615582743,0.083232
1.5821379598572,189.4856615582743,0.081608
1.581844106949773,189.5208615582743,0.08
1.5815503631775776,189.5560615582743,0.078408
1.581256728479827,189.5912615582743,0.076832
1.580963202795779,189.62646155827431,0.075272
1.580669786064737,189.66166155827432,0.073728
1.5803764782260494,189.69686155827432,0.0722
1.5800832792191093,189.7320615582743,0.070688
1.5797901889833554,189.7672615582743,0.069192
1.5794972074582705,189.8024615582743,0.067712
1.5792043345833828,189.8376615582743,0.066248
1.5789115702982652,189.8728615582743,0.0648
1.5786189145425353,189.9080615582743,0.063368
1.578326367255856,189.94326155827432,0.061952
1.5780339283779343,189.97846155827432,0.060552
1.577741597848522,190.01366155827432,0.059168
1.5774493756074157,190.0488615582743,0.0578
1.5771572615944565,190.0840615582743,0.056448
1.57686525574953,190.1192615582743,0.055112
1.5765733580125663,190.1544615582743,0.053792
1.5762815683235405,190.1896615582743,0.052488
1.575989886622471,190.2248615582743,0.0512
1.575698312849422,190.26006155827432,0.049928
1.5754068469445008,190.29526155827432,0.048672
1.5751154888478598,190.33046155827432,0.047432
1.574824238499695,190.3656615582743,0.046208
1.5745330958402475,190.4008615582743,0.045
1.5742420608098016,190.4360615582743,0.043808
1.5739511333486869,190.4712615582743,0.042632
1.5736603133972755,190.5064615582743,0.041472
1.5733696008959854,190.5416615582743,0.040328
1.5730789957852773,190.57686155827432,0.0392
1.5727884980056566,190.61206155827432,0.038088
1.5724981074976718,190.6472615582743,0.036992
1.5722078242019164,190.6824615582743,0.035912
1.5719176480590267,190.7176615582743,0.034848
1.5716275790096836,190.7528615582743,0.0338
1.5713376169946114,190.7880615582743,0.032768
1.5710477619545784,190.8232615582743,0.031752
1.5707580138303963,190.8584615582743,0.030752
1.5704683725629207,190.89366155827432,0.029768
1.5701788380930504,190.92886155827432,0.0288
1.569889410361728,190.9640615582743,0.027848
1.5696000893099404,190.9992615582743,0.026912
1.5693108748787163,191.0344615582743,0.025992
1.56902176700913,191.0696615582743,0.025088
1.5687327656422974,191.1048615582743,0.0242
1.5684438707193782,191.1400615582743,0.023328
1.5681550821815764,191.17526155827431,0.022472
1.567866399970138,191.21046155827432,0.021632
1.567577824026353,191.24566155827432,0.020808
1.5672893542915547,191.2808615582743,0.02
1.567000990707119,191.3160615582743,0.019208
1.5667127332144652,191.3512615582743,0.018432
1.5664245817550562,191.3864615582743,0.017672
1.5661365362703974,191.4216615582743,0.016928
1.565848596702037,191.4568615582743,0.0162
1.5655607629915667,191.49206155827432,0.015488
1.565273035080621,191.52726155827432,0.014792
1.5649854129108773,191.56246155827432,0.014112
1.5646978964240559,191.5976615582743,0.013448
1.5644104855619194,191.6328615582743,0.0128
1.564123180266274,191.6680615582743,0.012168
1.563835980478968,191.7032615582743,0.011552
1.5635488861418931,191.7384615582743,0.010952
1.563261897196983,191.7736615582743,0.010368
1.562975013586214,191.80886155827432,0.0098
1.5626882352516054,191.84406155827432,0.009248
1.5624015621352187,191.87926155827432,0.008712
1.5621149941791586,191.9144615582743,0.008192
1.561828531325571,191.9496615582743,0.007688
1.561542173516646,191.9848615582743,0.0072
1.561255920694614,192.0200615582743,0.006728
1.5609697728017493,192.0552615582743,0.006272
1.5606837297803682,192.0904615582743,0.005832
1.5603977915728289,192.12566155827432,0.005408
1.5601119581215321,192.16086155827432,0.005
1.559826229368921,192.1960615582743,0.004608
1.5595406052574796,192.2312615582743,0.004232
1.5592550857297358,192.2664615582743,0.003872
1.558969670728259,192.3016615582743,0.003528
1.5586843601956597,192.3368615582743,0.0032
1.5583991540745918,192.3720615582743,0.002888
1.5581140523077504,192.4072615582743,0.002592
1.5578290548378722,192.44246155827432,0.002312
1.5575441616077363,192.47766155827432,0.002048
1.5572593725601642,192.5128615582743,0.0018
1.556974687638018,192.5480615582743,0.001568
1.5566901067842023,192.5832615582743,0.001352
1.5564056299416633,192.6184615582743,0.001152
1.556121257053389,192.6536615582743,0.000968
1.5558369880624088,192.6888615582743,0.0008
1.555552822911794,192.72406155827431,0.000648
1.5552687615446574,192.75926155827432,0.000512
1.5549848039041532,192.79446155827432,0.000392
1.5547009499334772,192.8296615582743,0.000288
1.5544171995758667,192.8648615582743,0.0002
1.5541335527746005,192.9000615582743,0.000128
1.5538500094729986,192.9352615582743,7.2e-05
1.5535665696144225,192.9704615582743,3.2e-05
1.553283233142275,193.0056615582743,8e-06
1.553,193.04086155827432,0.0
1.552716870131083,193.07606155827432,8e-06
1.5524338434790506,193.11126155827432,3.2e-05
1.5521509199874701,193.1464615582743,7.2e-05
1.5518680995999505,193.1816615582743,0.000128
1.551585382260142,193.2168615582743,0.0002
1.5513027679117353,193.2520615582743,0.000288
1.5510202564984623,193.2872615582743,0.000392
1.550737847964096,193.3224615582743,0.000512
1.5504555422524504,193.35766155827432,0.000648
1.5501733393073804,193.39286155827432,0.0008
1.5498912390727815,193.42806155827432,0.000968
1.54960924149259,193.4632615582743,0.001152
1.5493273465107835,193.4984615582743,0.001352
1.5490455540713801,193.5336615582743,0.001568
1.5487638641184387,193.5688615582743,0.0018
1.5484822765960582,193.6040615582743,0.002048
1.5482007914483793,193.6392615582743,0.002312
1.5479194086195822,193.67446155827432,0.002592
1.5476381280538887,193.70966155827432,0.002888
1.5473569496955606,193.7448615582743,0.0032
1.5470758734888999,193.7800615582743,0.003528
1.5467948993782492,193.8152615582743,0.003872
1.5465140273079925,193.8504615582743,0.004232
1.546233257222553,193.8856615582743,0.004608
1.5459525890663943,193.9208615582743,0.005
1.5456720227840213,193.9560615582743,0.005408
1.545391558319978,193.99126155827432,0.005832
1.5451111956188497,194.02646155827432,0.006272
1.544830934625261,194.0616615582743,0.006728
1.5445507752838774,194.0968615582743,0.0072
1.544270717539404,194.1320615582743,0.007688
1.5439907613365862,194.1672615582743,0.008192
1.5437109066202095,194.2024615582743,0.008712
1.5434311533350993,194.2376615582743,0.009248
1.5431515014261215,194.27286155827431,0.0098
1.5428719508381807,194.30806155827432,0.010368
1.542592501516223,194.34326155827432,0.010952
1.5423131534052335,194.3784615582743,0.011552
1.5420339064502369,194.4136615582743,0.012168
1.5417547605962985,194.4488615582743,0.0128
1.5414757157885228,194.4840615582743,0.013448
1.5411967719720538,194.5192615582743,0.014112
1.5409179290920763,194.5544615582743,0.014792
1.5406391870938134,194.58966155827432,0.015488
1.540360545922529,194.62486155827432,0.0162
1.5400820055235254,194.66006155827432,0.016928
1.5398035658421456,194.6952615582743,0.017672
1.5395252268237716,194.7304615582743,0.018432
1.539246988413825,194.7656615582743,0.019208
1.5389688505577663,194.8008615582743,0.02
1.5386908132010964,194.8360615582743,0.020808
1.538412876289355,194.8712615582743,0.021632
1.5381350397681208,194.90646155827432,0.022472
1.537857303583013,194.94166155827432,0.023328
1.5375796676796882,194.97686155827432,0.0242
1.5373021320038442,195.0120615582743,0.025088
1.5370246965012166,195.0472615582743,0.025992
1.536747361117581,195.0824615582743,0.026912
1.5364701257987519,195.1176615582743,0.027848
1.5361929904905822,195.1528615582743,0.0288
1.535915955138965,195.1880615582743,0.029768
1.5356390196898317,195.22326155827432,0.030752
1.5353621840891531,195.25846155827432,0.031752
1.5350854482829386,195.2936615582743,0.032768
1.5348088122172363,195.3288615582743,0.0338
1.534532275838134,195.3640615582743,0.034848
1.5342558390917578,195.3992615582743,0.035912
1.5339795019242726,195.4344615582743,0.036992
1.533703264281882,195.4696615582743,0.038088
1.5334271261108288,195.5048615582743,0.0392
1.533151087357394,195.54006155827432,0.040328
1.5328751479678975,195.57526155827432,0.041472
1.5325993078886981,195.6104615582743,0.042632
1.5323235670661928,195.6456615582743,0.043808
1.5320479254468173,195.6808615582743,0.045
1.5317723829770455,195.7160615582743,0.046208
1.5314969396033908,195.7512615582743,0.047432
1.531221595272404,195.7864615582743,0.048672
1.5309463499306748,195.82166155827431,0.049928
1.530671203524831,195.85686155827432,0.0512
1.5303961560015398,195.89206155827432,0.052488
1.530121207307505,195.9272615582743,0.053792
1.52984635738947,195.9624615582743,0.055112
1.529571606194216,195.9976615582743,0.056448
1.5292969536685626,196.0328615582743,0.0578
1.5290223997593675,196.0680615582743,0.059168
1.5287479444135266,196.1032615582743,0.060552
1.5284735875779736,196.13846155827432,0.061952
1.5281993291996807,196.17366155827432,0.063368
1.527925169225658,196.20886155827432,0.0648
1.5276511076029538,196.2440615582743,0.066248
1.5273771442786541,196.2792615582743,0.067712
1.5271032791998824,196.3144615582743,0.069192
1.5268295123138018,196.3496615582743,0.070688
1.5265558435676114,196.3848615582743,0.0722
1.5262822729085488,196.4200615582743,0.073728
1.52600880028389,196.45526155827432,0.075272
1.525735425640948,196.49046155827432,0.076832
1.525462148927074,196.52566155827432,0.078408
1.5251889700896568,196.5608615582743,0.08
'''

class PycombsApp:
    def __init__(self):
        self.fig = None
        self.ui = None
        self.st = None
        self.ani = None

APP = None

class LLEState:
    
    def __init__(self):
        '''Constants'''

        self.hbar = 1.0545718e-34  # [J*s]
        self.c = 299792458         # [m/s]
        self.i = 1j                # Imag unit

        '''Pump light'''

        self.wvl_pump = 1553e-9                      # [m]
        self.frq_pump = self.c / self.wvl_pump       # [Hz]
        self.ome_pump = 2 * np.pi * self.frq_pump    # [rad/s]

        '''Dispersion input'''

        self.dint_file_choice = True
        self.dint_file_path = r"dint_mgf2.csv"
        self.d2_file_path = "700w_d2_output.txt"
        self.d2_default = 50  # [rad/s]  (check units / meaning)

        '''Resonator parameters'''

        self.R_res = (27e-6) / 2                # [m]
        self.L = self.R_res * np.pi * 2         # [m]
        self.fsr = 35.2e9                         # [Hz]
        self.Q = 4e8                          # [-]

        '''Material parameters and nonlinearity'''

        self.n2 = 1.1e-20       # [m^2/W] (check your chosen units)
        self.gamma_v = 2.4e2    # [1/(W*m)] (check)
        self.Aeff = 1.6 * ((1e-6)**2) # [um^2]
        
        '''Power and coupling'''

        self.P_in_phys = None # 9.3e-3                 # [W]
        self.P_in_norm_default = [12]         # [-] (keep list for now)
        self.eta = 0.5                         # [-] coupling efficiency
        # self.kappa_ex will be derived below

        '''Detuning start and stop'''

        self.Detuning_normalized_start = -4
        self.Detuning_normalized_stop = 15
        self.detuning_sweep_rate = 500000

        '''Simulation parameters'''

        self.number_modes = 200
        self.save_step_point = 2000
        self.plot_step = 5000
        
        '''Noise control'''

        '''Noise switches'''

        self.noise_switch = True
        self.pump_noise_enabled = True
        self.cavity_noise_enabled = True

        '''Derived physical parameters'''

        self.t_phys_round_trip = 1 / self.fsr                 # [s]
        self.linewidth = self.frq_pump / self.Q               # [Hz]
        self.kappa_avg = self.linewidth * 2 * np.pi           # [rad/s]
        self.alpha = self.t_phys_round_trip * self.kappa_avg / 2
        self.norm_t = 1 / (self.alpha / self.t_phys_round_trip)
        self.kappa_ex = self.eta * self.kappa_avg

        self.mu = np.arange(-self.number_modes // 2, self.number_modes // 2)
        self.frq_grid = self.frq_pump + self.mu * self.fsr
        self.ome_grid = self.frq_grid * (2 * np.pi)
        self.wvl_grid = self.c / self.frq_grid

        '''Definitions of time and integration time for the simulation'''

        self.round_trips_per_integration = 1
        self.tal_step = self.round_trips_per_integration * self.t_phys_round_trip / self.norm_t
        
        #this bool enables cubic interpolation on the temporal plot
        #doesn't seem that useful but the wiggles don't seem to be artifacts if num modes cranked in comparison
        self.temporal_interpol = False 

        '''Dispersion file loading'''

        self._load_dispersion_with_fallback()


        '''Avoided mode crossing (AMX)'''

        self.AMX_strength = 0
        self.AMX_loc = 72.5
        self.AMX_epsilon = 1e-6
        self.AMX_Lorentzian_width = 1.0

        self.AMX = -self.AMX_strength / (((self.mu - self.AMX_loc) + self.AMX_epsilon)**2 + self.AMX_Lorentzian_width**2)
        self.dint = self.dint + self.AMX

        '''Normalized integrated dispersion'''

        self.dint_norm = 2 * self.dint / self.kappa_avg
        self.kappa_all = np.ones(self.number_modes) * self.kappa_avg

        '''Derived physical coefficients'''

        self.ng0 = self.c / (self.fsr * self.L)

        # self.Aeff = self.n2 * (2 * np.pi * self.frq_pump) / (self.c * self.gamma_v)  # alt definition

        self.veff = self.Aeff * self.L
        self.g = self.hbar * (2 * np.pi * self.frq_pump)**2 * self.c * self.n2 / (self.ng0**2 * self.veff)

        self.E_amp_2_norm_factor = np.sqrt(self.kappa_avg / (2 * self.g))
        self.P_in_norm_factor = (self.hbar * self.ome_pump * self.kappa_avg**2) / (8 * self.g * self.eta)

        '''Choice of physical input or normalized input'''

        if self.P_in_phys is not None:
            self.P_in_norm = [self.P_in_phys / self.P_in_norm_factor]
            print(f"Using physical input power: {self.P_in_phys*1e6:.1f} µW (P_in_norm = {self.P_in_norm[0]:.3f})")
        else:
            self.P_in_norm = self.P_in_norm_default
            print(f"Using default normalized input power: {self.P_in_norm[0]}")

        self.noise_amp = np.sqrt(1.0 / (2.0 * self.tal_step)) / self.E_amp_2_norm_factor
        self.pump_noise_amp = self.noise_amp
        self.sig_noise_amp = self.noise_amp

        # ---- GUI version: pick one input power value (no loop) ----
        self.P_norm = float(self.P_in_norm[0])     # canonical variable
        self.P_norm_target = self.P_norm
        
        self.S = np.sqrt(self.P_norm)
        self.S_target = self.S

        # Detuning range is still useful for slider limits / labels
        self.detuning_norm_sweep_start = self.Detuning_normalized_start
        self.detuning_norm_sweep_stop = self.Detuning_normalized_stop

        # These were batch-sweep bookkeeping; keep if you still want them for reference
        self.t_norm_total = round(
            self.detuning_sweep_rate * (-self.detuning_norm_sweep_start + self.detuning_norm_sweep_stop) / self.save_step_point
        ) * self.save_step_point
        self.t_phys_total = (2 / self.kappa_avg) * self.t_norm_total
        self.N_iter = int(self.t_norm_total / self.round_trips_per_integration)

        self.delta_detuning_norm_int_step = (self.detuning_norm_sweep_stop - self.detuning_norm_sweep_start) / max(self.N_iter, 1)

        # In GUI, DV is controlled by slider:
        self.DV = self.detuning_norm_sweep_start
        self.target_detuning = self.DV
        #st.detuning_slew = 0.5 / steps_per_frame
        # Slew rate in DV per nanosecond (DV/ns)
        self.detuning_slew_rate = 0.01   # DV/ns (pick a reasonable default)
        self.dt_phys_s  = (2.0 / self.kappa_avg) * self.tal_step
        self.dt_phys_ns = self.dt_phys_s * 1e9

        '''Input field (CW by default, optional file)'''

        self.input_field_file = "input_field_time.txt"

        if os.path.isfile(self.input_field_file):
            data = np.loadtxt(self.input_field_file, skiprows=1)
            input_field_complex = data[:, 1] + 1j * data[:, 2]

            if len(input_field_complex) != self.number_modes:
                raise ValueError(f"Input field length mismatch: expected {self.number_modes}, got {len(input_field_complex)}")

            self.tE_in = input_field_complex
            self.E_in = fft(self.tE_in)
            self.fE_in_o = fftshift(self.E_in)
            print("Loaded custom input field from file.")
        else:
            self.tE_in = np.full(self.mu.shape, self.S)
            self.E_in = fft(self.tE_in)
            self.fE_in_o = fftshift(self.E_in)
            print("Using default CW input field.")

        # --- CW pump + vacuum-noise cavity startup ---
        # --- Reference-style noisy pump + noisy initial cavity ---
        
        # noisy pump spectrum
        self.fE_in = self.fE_in_o + self._complex_noise(self.pump_noise_amp, self.number_modes)
        
        # noisy initial intracavity field
        self.E_t_fast_norm = self._complex_noise(self.sig_noise_amp, self.number_modes)
        
        # initial intracavity spectrum
        self.spectrum_E_t_fast_norm = fftshift(fft(self.E_t_fast_norm))
        
        # half-step pump injection
        self.input_pump_half_step = self.fE_in * (self.tal_step / 2)


        '''Fast time axis'''

        self.tau_phys = np.linspace(-self.t_phys_round_trip / 2, self.t_phys_round_trip / 2, (self.number_modes))
        self.tau_ps = self.tau_phys * 1e12
        self.tau_fs = self.tau_phys * 1e15
        self.tau = self.tau_phys

        self.j = 0
        self.pump_idx = np.where(self.mu == 0)[0][0]

        # Optional convenience (so you can use kappa_all/kappa_avg in one array)
        self.kappa_all_over_kappa_avg = self.kappa_all / self.kappa_avg

    def set_targets(self, detuning=None, slew=None, P_norm=None):
        if detuning is not None:
            self.target_detuning = float(detuning)
            
        if slew is not None:
            self.target_slew = float(slew)
    
        if P_norm is not None:
            self.P_norm_target = max(float(P_norm), 0.0)
            self.S_target = np.sqrt(self.P_norm_target)
            
    def _complex_noise(self, amp, size, kind="cavity"):
        # master switch overrides everything
        if not self.noise_switch:
            return np.zeros(size, dtype=complex)
    
        if kind == "pump" and not self.pump_noise_enabled:
            return np.zeros(size, dtype=complex)
    
        if kind == "cavity" and not self.cavity_noise_enabled:
            return np.zeros(size, dtype=complex)
    
        return amp * (np.random.randn(size) + 1j*np.random.randn(size))


    def step(self, n_steps=100):
        for _ in range(n_steps):
            self.j += 1
    
            # --- physical time per internal step ---
            dt_phys = (2.0 / self.kappa_avg) * self.tal_step   # seconds
            dt_phys_ns = dt_phys * 1e9                         # ns
    
            # --- slew-limited detuning update ---
            max_DV_step = self.detuning_slew_rate * dt_phys_ns
            d = self.target_detuning - self.DV
            self.DV += np.clip(d, -max_DV_step, max_DV_step)
    
            # --- power update: rebuild clean CW pump, then noisy pump ---
            if self.P_norm_target != self.P_norm:
                self.S = self.S_target
                self.P_norm = self.P_norm_target
            
                self.tE_in = np.full(self.mu.shape, self.S)
                self.E_in = fft(self.tE_in)
                self.fE_in_o = fftshift(self.E_in)
            
                self.fE_in = self.fE_in_o + self._complex_noise(self.pump_noise_amp, self.number_modes)
                self.input_pump_half_step = self.fE_in * (self.tal_step / 2)
            
            # --- refresh pump noise occasionally, like reference code ---
            if self.j % self.save_step_point == 0:
                self.fE_in = self.fE_in_o + self._complex_noise(self.pump_noise_amp, self.number_modes)
                self.input_pump_half_step = self.fE_in * (self.tal_step / 2)
            
            # --- propagation constants ---
            L = -(self.kappa_all / self.kappa_avg) - 1j * self.DV - 1j * self.dint_norm
            dt = self.tal_step
            h = dt / 2.0
            exp_prop = np.exp(L * h)
            
            # first linear half-step
            self.spectrum_E_t_fast_norm = exp_prop * (
                self.spectrum_E_t_fast_norm + self.input_pump_half_step
            )
            
            # nonlinear step
            self.E_t_fast_norm = ifft(fftshift(self.spectrum_E_t_fast_norm))
            self.E_t_fast_norm *= np.exp(1j * (np.abs(self.E_t_fast_norm) ** 2) * dt)
            
            # second linear half-step
            self.spectrum_E_t_fast_norm = fftshift(fft(self.E_t_fast_norm))
            self.spectrum_E_t_fast_norm = exp_prop * (
                self.spectrum_E_t_fast_norm + self.input_pump_half_step
            )

    def get_spectrum_dbm_like(self):
        P = np.abs(self.spectrum_E_t_fast_norm)**2
        P = np.maximum(P, 1e-30)  # safer floor
        return 10*np.log10(P)
    
    def get_output_spectrum_dBm(self):
        """
        OSA-like spectrum at the through port (bus waveguide), per mode, in dBm.
    
        Uses:
          - A_mu_norm from current spectrum_E_t_fast_norm
          - a_mu_phys = E_amp_2_norm_factor * A_mu_norm
          - s_out = s_in - sqrt(kappa_ex) * a_mu_phys
          - P_out = hbar * omega_mu * |s_out|^2  [W]
        """
        # Convert normalized spectral field -> normalized mode amplitudes.
        # NOTE: your FFT conventions vary; this /number_modes matches your older batch code.
        A_mu_norm = self.spectrum_E_t_fast_norm / self.number_modes
    
        # Physical mode amplitudes
        a_mu_phys = self.E_amp_2_norm_factor * A_mu_norm
    
        # Physical pump power corresponding to current normalized pump power
        P_in_phys_W = self.P_norm * self.P_in_norm_factor
    
        # Build s_in (only pump mode is driven)
        s_in = np.zeros_like(a_mu_phys, dtype=complex)
        s_in[self.pump_idx] = np.sqrt(max(P_in_phys_W, 0.0) / (self.hbar * self.ome_pump))
    
        # Through port field
        s_out = s_in - np.sqrt(self.kappa_ex) * a_mu_phys
    
        # Output power per mode [W]
        P_out_W = self.hbar * self.ome_grid * (np.abs(s_out) ** 2)
    
        # Convert to dBm safely
        P_out_W = np.maximum(P_out_W, 1e-30)
        P_out_dBm = 10.0 * np.log10(P_out_W / 1e-3)
    
        return P_out_dBm


    def get_intracavity_mean(self):
        return float(np.mean(np.abs(self.E_t_fast_norm)**2))
    
    def get_intracavity_mean_W(self):
        """
        Approx physical mean intracavity power in W.
        Uses the same normalization logic as your per-mode formula:
          a = E_amp_2_norm_factor * A
          P ≈ sum(hbar*omega*|a|^2) / N
        """
        A = self.E_t_fast_norm                       # normalized field vs fast time
        a = self.E_amp_2_norm_factor * A             # physical amplitude (per your normalization)
        P_inst = self.hbar * self.ome_pump * np.abs(a)**2  # W-like, using pump omega as approx
        return float(np.mean(P_inst))
    
    def reinitialize(self, new_n_modes=None):
        """Recalculates physical constants and resets the simulation state."""
        if new_n_modes is not None:
            self.number_modes = int(new_n_modes)
            self.mu = np.arange(-self.number_modes // 2, self.number_modes // 2)
            self.kappa_all = np.ones(self.number_modes) * self.kappa_avg
        
        # --- REBUILD TIME NORMALIZATION (missing right now) ---
        # Make sure pump frequency is consistent with current wavelength
        self.frq_pump = self.c / self.wvl_pump
        self.ome_pump = 2 * np.pi * self.frq_pump
        
        self.t_phys_round_trip = 1.0 / self.fsr
        self.linewidth = self.frq_pump / self.Q
        self.kappa_avg = self.linewidth * 2 * np.pi
        self.kappa_ex  = self.eta * self.kappa_avg
        
        # normalized time definitions (same as __init__)
        self.alpha  = self.t_phys_round_trip * self.kappa_avg / 2.0
        self.norm_t = 1.0 / (self.alpha / self.t_phys_round_trip)
        
        self.round_trips_per_integration = 1
        self.tal_step = self.round_trips_per_integration * self.t_phys_round_trip / self.norm_t
        
        # update dt used by DV/ns logic
        self.dt_phys_s  = (2.0 / self.kappa_avg) * self.tal_step
        self.dt_phys_ns = self.dt_phys_s * 1e9

        
        # Recalculate Nonlinearity g and normalization factors
        self.L = self.R_res * np.pi * 2
        self.veff = self.Aeff * self.L
        self.ng0 = self.c / (self.fsr * self.L)
        self.g = self.hbar * (2 * np.pi * self.frq_pump)**2 * self.c * self.n2 / (self.ng0**2 * self.veff)
        self.E_amp_2_norm_factor = np.sqrt(self.kappa_avg / (2 * self.g))
        self.P_in_norm_factor = (self.hbar * self.ome_pump * self.kappa_avg**2) / (8 * self.g * self.eta)

        self.noise_amp = np.sqrt(1.0 / (2.0 * self.tal_step)) / self.E_amp_2_norm_factor
        self.pump_noise_amp = self.noise_amp
        self.sig_noise_amp = self.noise_amp

        # --- KEEP PHYSICAL INPUT POWER CONSTANT WHEN WAVELENGTH CHANGES ---
        if self.P_in_phys is not None:
            self.P_in_norm = [self.P_in_phys / self.P_in_norm_factor]
        else:
            self.P_in_norm = self.P_in_norm_default

        # canonical GUI variables
        self.P_norm = float(self.P_in_norm[0])
        self.P_norm_target = self.P_norm
        self.S = np.sqrt(self.P_norm)
        self.S_target = self.S


        # Update grids
        self.frq_grid = self.frq_pump + self.mu * self.fsr
        self.ome_grid = self.frq_grid * (2 * np.pi)
        self.wvl_grid = self.c / self.frq_grid
        
        self._load_dispersion_with_fallback()
        self.dint_norm = 2 * self.dint / self.kappa_avg


        self.dint_norm = 2 * self.dint / self.kappa_avg
        
        # Reset state
        self.j = 0
        
        # re-seed cavity with small vacuum-like noise
        # reset to empty cavity; stochastic drive will seed the comb
        # noisy initial intracavity field
        self.E_t_fast_norm = self._complex_noise(self.sig_noise_amp, self.number_modes)
        
        # rebuild spectrum from noisy cavity
        self.spectrum_E_t_fast_norm = fftshift(fft(self.E_t_fast_norm))
        
        self.tau_phys = np.linspace(-self.t_phys_round_trip/2, self.t_phys_round_trip/2, self.number_modes)
        self.tau_ps = self.tau_phys * 1e12
        
        self.pump_idx = np.where(self.mu == 0)[0][0]        
        
        # --- REBUILD CW INPUT FIELD + INJECTION TERM (consistent with __init__) ---
        self.tE_in = np.full(self.mu.shape, self.S)
        self.E_in = fft(self.tE_in)
        self.fE_in_o = fftshift(self.E_in)
        
        # noisy pump spectrum
        self.fE_in = self.fE_in_o + self._complex_noise(self.pump_noise_amp, self.number_modes)
        
        self.input_pump_half_step = self.fE_in * (self.tal_step / 2)

        
        # Ensure the spectral field is exactly the right shape
        self.spectrum_E_t_fast_norm = fftshift(fft(self.E_t_fast_norm))
        
        self.kappa_all = np.ones(self.number_modes) * self.kappa_avg
        self.kappa_all_over_kappa_avg = self.kappa_all / self.kappa_avg

    def _load_dispersion_with_fallback(self):
        """
        Sets:
          self.dint, self.d2, and prints what was used.
        Tries:
          (A) self.dint_file_path if valid
          (B) embedded DEFAULT_DINT_CSV
          (C) constant d2_default
        """
        # Treat blank strings as "not provided"
        path = (self.dint_file_path or "").strip()
    
        # -------------- (A) try file --------------
        if path:
            # also try relative-to-script directory for convenience
            candidates = [path]
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                candidates.append(os.path.join(base_dir, path))
            except NameError:
                pass  # __file__ may not exist in some environments
    
            file_found = next((p for p in candidates if os.path.isfile(p)), None)
            if file_found:
                dint_data = np.loadtxt(file_found, skiprows=1, delimiter=',')
                self._build_dint_from_table(dint_data)
                print(f'Using Dint file: "{file_found}"')
                return
            else:
                print(f'[Dispersion] Dint file not found: "{path}" -> using embedded default.')
    
        # -------------- (B) embedded default --------------
        try:
            dint_data = np.genfromtxt(io.StringIO(DEFAULT_DINT_CSV), delimiter=',', skip_header=1)
            if dint_data.ndim == 1:
                dint_data = dint_data.reshape(1, -1)
            self._build_dint_from_table(dint_data)
            print('[Dispersion] Using embedded default Dint.')
            return
        except Exception as e:
            print(f"[Dispersion] Embedded default failed ({e!r}) -> using constant d2_default.")
    
        # -------------- (C) constant D2 fallback --------------
        self.d2 = self.d2_default * np.ones_like(self.mu)
        self.dint = 0.5 * self.d2 * self.mu**2
        print(f"[Dispersion] Using constant d2_default = {self.d2_default}")
    
    def _build_dint_from_table(self, dint_data):
        """
        dint_data columns expected:
          col0: wavelength [um]
          col1: frequency [THz]
          col2: (dint/2pi) [GHz]
        """
        self.dint_wvl = dint_data[:, 0] * 1e-6
        self.dint_frq = dint_data[:, 1] * 1e12
        self.dint_ome = self.dint_frq * (2 * np.pi)
    
        self.dint_vals = dint_data[:, 2] * 1e9              # (dint/2pi) in Hz
        self.dint_vals_ome = self.dint_vals * 2 * np.pi      # [rad/s]
    
        dint_interp = interp1d(self.dint_ome, self.dint_vals_ome, kind='cubic', fill_value="extrapolate")
        self.dint = dint_interp(self.ome_grid)               # [rad/s]
    
        # Your existing d2 extraction (keep as-is)
        self.d2_ome = np.zeros_like(self.dint)
        mask = (self.mu != 0)
        self.d2_ome[mask] = self.dint[mask] * 2 / (self.mu[mask]**2)
        self.d2_frq = self.d2_ome / (2 * np.pi)
        self.d2 = self.d2_ome.copy()
        self.d2[self.mu == 0] = 0



def run_gui():
    plt.close("all")
    st = LLEState()
    # Create a persistent container for all interactive elements
    ui = {}
    
    DEFAULTS = dict(
        DV=st.DV,
        Pnorm=st.P_norm,
        slew=st.detuning_slew_rate,
    )
    
    # ---- Spectrum Y-axis mode ----
    SPECTRUM_LIVE_AUTOSCALE = False   # True = live autoscale, False = fixed limits
    
    # Fixed limits (only used if autoscale is False)
    SPECTRUM_YMIN = -300
    SPECTRUM_YMAX = 120
    baseline_db = -299
    
    # ---- Detuning limits (single source of truth) ----
    DV_min = -5.0
    DV_max = 15.0
    
    Pnorm_min = 0.0
    Pnorm_max = 20.0

    # ---- Live settings ----
    steps_per_frame = 100 #10 #50          # sim steps per animation frame
    Nkeep = 3000                  # how many power samples to keep (ring buffer)
    interval_ms = 100              # animation interval

    # ---- Figure ----
    fig = plt.figure(figsize=(13, 6))
    fig.canvas.manager.set_window_title("pycombs v1.1 half SSFM")
    
    # make window open maximized
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()


    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])
    
    ax_spec  = fig.add_subplot(gs[0, 0])   # spectrum (top-left)
    ax_pulse = fig.add_subplot(gs[1, 0])   # NEW: temporal pulse (bottom-left)
    ax_pow   = fig.add_subplot(gs[:, 1])   # power history (right, full height)
    
    # ---- pycomb label above spectrum plot ----
    pos = ax_spec.get_position()
    
    fig.text(
        pos.x0-0.1,                 # align with left of spectrum axis
        pos.y1 + 0.06,          # slightly above it
        "pycombs v9 half SSFM",
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color="black"
    )

    
    # Reduce right to 0.75 to give the textboxes more breathing room
    plt.subplots_adjust(
        bottom=0.25,
        #top=0.8,      # ← add this (moves plots downward)
        wspace=0.35,
        hspace=0.50,
        right=0.75
    )


    # ---- Spectrum plot (OSA-like) ----
    
    # --- Frequency axis (uniformly spaced, correct physics) ---
    x_THz = st.frq_grid * 1e-12   # main x-axis

    
    def wvl_nm_to_THz(wvl_nm):
        return st.c / (wvl_nm * 1e-9) * 1e-12
    
    def THz_to_wvl_nm(freq_THz):
        return st.c / (freq_THz * 1e12) * 1e9

    def wvl_um_to_THz(wvl_um):
        wvl_um = np.asarray(wvl_um, dtype=float)
        wvl_um = np.where(np.abs(wvl_um) < 1e-12, np.nan, wvl_um)
        return st.c / (wvl_um * 1e-6) * 1e-12


    def THz_to_wvl_um(freq_THz):
        return st.c / (freq_THz * 1e12) * 1e6

    
    def choose_baseline_db(spec_db, pump_idx, q=10, margin_db=10):
        tmp = np.array(spec_db, float)
        tmp[pump_idx] = np.nan
        floor = np.nanpercentile(tmp, q)
        return float(floor - margin_db)
    
    def apply_minor_ticks(ax):
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which="minor", length=3)

    
    # Initial OSA-like spectrum (sorted to match x_nm)
    y_db = st.get_output_spectrum_dBm()
    spec_line, = ax_spec.plot(x_THz, y_db, linewidth=0.0)

    ax_spec.set_title("Pulse spectrum", fontsize=13, fontweight="bold")    
    ax_spec.set_xlabel("Frequency (THz)")
    ax_spec.set_ylabel("Spectrum optical power\n(through-port) [dBm]")
    spec_color = "tab:blue"  # same as stem_sc/stem_lc
    ax_spec.yaxis.label.set_color(spec_color)
    ax_spec.yaxis.label.set_color(spec_color)
    ax_spec.grid(True, alpha=0.25)
    
    # Always match simulated-mode span
    ax_spec.set_xlim(x_THz[0], x_THz[-1])
    apply_minor_ticks(ax_spec)

    
    # Optional fixed Y limits
    if not SPECTRUM_LIVE_AUTOSCALE:
        ax_spec.set_ylim(SPECTRUM_YMIN, SPECTRUM_YMAX)
    
    # Secondary top axis: frequency THz
    ax_spec_top = ax_spec.secondary_xaxis(
        "top",
        functions=(THz_to_wvl_um, wvl_um_to_THz)
    )
    ax_spec_top.set_xlabel("Wavelength (µm)")
    
    ax_spec_top.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_spec_top.minorticks_on()
    ax_spec_top.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_spec_top.tick_params(which="minor", length=3)
    
    # Baseline for stems
    pump_idx = st.pump_idx
    #baseline_db = choose_baseline_db(y_db, pump_idx)
    baseline_state = {"y": baseline_db, "locked": True}
    baseline_line = ax_spec.axhline(baseline_state["y"], color="k", linestyle="--", linewidth=1)
    
    # Stems + dots in WAVELENGTH units (sorted)
    segments = [((x, baseline_state["y"]), (x, y)) for x, y in zip(x_THz, y_db)]
    stem_sc = ax_spec.scatter(x_THz, y_db, c="tab:blue", s=0)
    stem_lc = LineCollection(segments, colors="tab:blue", linewidths=1)
    ax_spec.add_collection(stem_lc)
    
    # ---- Temporal pulse plot (INIT ONCE) ----
    t_ps = st.tau_ps
    I_t  = np.abs(st.E_t_fast_norm)**2
    t_pulse = t_ps
    I_t_pulse = I_t
    
    def sync_x_axes_to_grid(st, ax_spec, ax_pulse, ax_spec_top=None, pad_frac=0.0):
        # --- Spectrum axis (THz) ---
        x_THz = st.frq_grid * 1e-12
        xmin, xmax = float(x_THz[0]), float(x_THz[-1])
    
        # optional small padding
        pad = (xmax - xmin) * pad_frac
        ax_spec.set_xlim(xmin - pad, xmax + pad)
    
        # --- Temporal axis (ps) ---
        t_ps = st.tau_ps
        tmin, tmax = float(t_ps[0]), float(t_ps[-1])
        pad_t = (tmax - tmin) * pad_frac
        ax_pulse.set_xlim(tmin - pad_t, tmax + pad_t)
    
        # If you use a secondary wavelength axis, it will follow xlim automatically,
        # but forcing a draw helps refresh tick formatting.
        if ax_spec_top is not None:
            ax_spec_top.figure.canvas.draw_idle()
    
        return x_THz, t_ps


    pulse_line, = ax_pulse.plot(t_pulse, I_t_pulse, linewidth=1.2, color="tab:blue")
    
    ax_pulse.set_title("Pulse intracavity temporal profile", fontsize=13,fontweight="bold")
    ax_pulse.set_xlabel("Fast time τ (ps)")
    ax_pulse.set_ylabel("Intracavity intensity\n(arb. units)")
    pulse_color = "tab:blue"
    ax_pulse.yaxis.label.set_color(pulse_color)
    ax_pulse.yaxis.label.set_color(pulse_color)
    ax_pulse.grid(True, alpha=0.3)
    ax_pulse.set_ylim(0, 25)
    apply_minor_ticks(ax_pulse)
    
    # Sync spectrum + temporal x axes to current N and FSR
    x_THz, t_ps = sync_x_axes_to_grid(st, ax_spec, ax_pulse, ax_spec_top, pad_frac=0.01)


    def DV_to_detuning_GHz(DV, kappa_avg):
        # detuning_Hz = DV * kappa/(2π)
        return DV * kappa_avg / (2*np.pi) * 1e-9

    def detuning_GHz_to_DV(det_GHz, kappa_avg):
        return det_GHz * 1e9 * (2*np.pi) / kappa_avg
    
    def Pnorm_to_W(Pnorm, Pnorm_factor):
        return Pnorm * Pnorm_factor

    def W_to_Pnorm(P_W, Pnorm_factor):
        return P_W / Pnorm_factor
    
    def S_to_W(S, Pnorm_factor):
        return (S**2) * Pnorm_factor
    
    def W_to_S(P_W, Pnorm_factor):
        return np.sqrt(max(P_W / Pnorm_factor, 0.0))

    def get_fast_time_axis(st):
        """
        Returns (x, xlabel) for the time-domain plot.
        Tries to use physical fast time in ps if available; otherwise uses index.
        """
        if hasattr(st, "tau_ps"):
            return st.tau_ps, "Fast time τ (ps)"
        if hasattr(st, "tau_phys"):
            return st.tau_phys * 1e12, "Fast time τ (ps)"
        if hasattr(st, "tau"):
            return st.tau, "Fast time τ (norm)"
        # fallback
        return np.arange(st.number_modes), "Sample index"

    # ---- Power plot (ring buffer): 3 TRACES ONLY (all normalized),
    #      with correlated physical axes (secondary axes, no extra traces) ----
    pow_x = np.full(Nkeep, np.nan, dtype=float)
    pow_i = {"i": 0}

    # Histories (ONLY 3 quantities)
    y_Pcav_W = np.full(Nkeep, np.nan, dtype=float)  # ⟨P_cav⟩ in W (this is the plotted line)
    y_DV     = np.full(Nkeep, np.nan, dtype=float)  # DV
    y_Pnorm  = np.full(Nkeep, np.nan, dtype=float)  # P_norm


    # ---------- styling knobs ----------
    FS_LABEL = 9
    FS_TICK  = 8
    LW_MAIN  = 1.2

    RIGHT_OFFSETS = [0, 50, 110]  # more spacing so tick labels don't overlap

    def _style_axis_right(ax, color, label, offset_pts=0):
        # Move spine outward
        ax.spines["right"].set_position(("outward", offset_pts))
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_visible(False)
    
        # Ticks & label on the right only
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    
        # Spine styling
        ax.spines["right"].set_linewidth(1.0)
        ax.spines["right"].set_color(color)
    
        # Tick styling
        ax.tick_params(
            axis="y",
            which="both",
            colors=color,
            labelsize=FS_TICK,
            pad=4,
            width=1.0,
            length=4
        )
    
        # Label styling
        ax.set_ylabel(label, fontsize=FS_LABEL, labelpad=10, color=color)
        ax.yaxis.label.set_color(color)
        
    # Secondary RIGHT axis = normalized intracavity power scale (mirrors the left axis)
    
    K_Pcav_mW = 1e3 * float(st.hbar * st.ome_pump * (st.E_amp_2_norm_factor**2)) / st.t_phys_round_trip

    def Pcav_mW_to_norm(PmW):
        return PmW / K_Pcav_mW
    
    def Pcav_norm_to_mW(Pn):
        return Pn * K_Pcav_mW


    # Base axis: normalized intracavity mean ⟨|E|²⟩ (RIGHT side)
    ax_pow.set_title("Intracavity power evolution", fontsize=13, fontweight="bold")
    ax_pow.set_xlabel("Time (ns)")
    ax_pow.grid(True, alpha=0.25)
    ax_pow.set_ylim(Pcav_norm_to_mW(0), Pcav_norm_to_mW(10))
    apply_minor_ticks(ax_pow)

    
    hud = ax_pow.text(
        0.02, 0.98, "",                 # (x,y) in axes fraction
        transform=ax_pow.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.85)
    )



    # Base axis = PHYSICAL intracavity power (LEFT)
    ax_pow.set_ylabel(
        "Mean physical intracavity power [mW]",
        fontsize=FS_LABEL,
        color="tab:blue"
    )
    
    ax_pow.tick_params(
        axis="y",
        labelsize=FS_TICK,
        pad=4,
        width=1.0,
        length=4,
        colors="tab:blue"
    )
    
    ax_pow.spines["left"].set_visible(True)
    ax_pow.spines["left"].set_color("tab:blue")
    ax_pow.spines["left"].set_linewidth(1.0)

    ax_pow.spines["right"].set_visible(False)



    
    ax_pow_norm = ax_pow.secondary_yaxis(
        "right",
        functions=(Pcav_mW_to_norm, Pcav_norm_to_mW)
    )
    ax_pow_norm.set_ylabel("Mean normalized intracavity power [norm]", fontsize=FS_LABEL, color="tab:blue")
    ax_pow_norm.tick_params(axis="y", which="both", colors="tab:blue", labelsize=FS_TICK, pad=4, width=1.0, length=4)
    ax_pow_norm.spines["right"].set_color("tab:blue")
    ax_pow_norm.spines["right"].set_linewidth(1.0)
    
    ax_pow_norm.spines["right"].set_position(("outward", RIGHT_OFFSETS[0]))  # = 0, explicit
    ax_pow_norm.set_zorder(10)  # keep it on top of base patch
    apply_minor_ticks(ax_pow_norm)


    # ----- Extra RIGHT axes for DV and P_norm (normalized), still only 3 traces total -----
    ax_pow_R2 = ax_pow.twinx()
    _style_axis_right(ax_pow_R2, "tab:green", "Normalized pump detuning value [norm]", offset_pts=RIGHT_OFFSETS[1])
    ax_pow_R2.set_ylim(DV_min, DV_max)
    apply_minor_ticks(ax_pow_R2)
    
    ax_pow_R3 = ax_pow.twinx()
    _style_axis_right(ax_pow_R3, "tab:purple", "Normalized pump input power [norm]", offset_pts=RIGHT_OFFSETS[2])
    ax_pow_R3.set_ylim(Pnorm_min, Pnorm_max)
    apply_minor_ticks( ax_pow_R3) 

    # Lines (3 TOTAL)
    line_Pcav_W, = ax_pow.plot([], [], color="tab:blue", linewidth=LW_MAIN)
    line_DV,        = ax_pow_R2.plot([], [], color="tab:green", linewidth=LW_MAIN)
    line_Pnorm,     = ax_pow_R3.plot([], [], color="tab:purple", linewidth=LW_MAIN)

    ax_pow.tick_params(axis="x", labelsize=FS_TICK, pad=2, width=0.8, length=3)


    # # ---- Sliders ----
    
    # ---- Sliders ----
    
    slider_spacing = 0.024
    det_slider_bottom = 0.145
    slider_height = 0.030
    slider_width = 0.7
    
    def textbox_next_to_slider(ax_slider, width=0.05, height=0.03, gap=0.01):
        pos = ax_slider.get_position()
        return plt.axes([pos.x1 + gap, pos.y0, width, height])


    def add_slider_ticks_inside_step(ax, slider, step=1.0, y0=0.25, y1=0.75, lw=0.8, color="0.25"):
        vmin, vmax = float(slider.valmin), float(slider.valmax)
    
        start = np.ceil(vmin / step) * step
        xs = np.arange(start, vmax + 0.5*step, step)
    
        trans = ax.get_xaxis_transform()  # x=data, y=axes
        lines = []
        for x in xs:
            lines.append(
                ax.plot([x, x], [y0, y1], transform=trans,
                        clip_on=True, linewidth=lw, color=color, zorder=5)[0]
            )
        return lines

    
    ax_det = plt.axes([0.12, det_slider_bottom, slider_width ,slider_height])
    det_slider = Slider(ax_det, "Detuning (DV)", DV_min, DV_max, valinit=st.DV, valfmt='%.3f')
    ui["det_inside_ticks"] = add_slider_ticks_inside_step(ax_det, det_slider, step=1.0)

    det_init_GHz = DV_to_detuning_GHz(st.DV, st.kappa_avg)
    
    
    det_slider.valtext.set_visible(False)
    
    ax_det_text = textbox_next_to_slider(ax_det)
    det_text_DV = TextBox(ax_det_text,'',initial=f"{st.DV:.3f}",color='0.95', hovercolor='0.95')
    
    
    det_min_GHz = DV_to_detuning_GHz(DV_min, st.kappa_avg)
    det_max_GHz = DV_to_detuning_GHz(DV_max, st.kappa_avg)
    det_init_GHz = DV_to_detuning_GHz(st.DV, st.kappa_avg)

    
    ax_detGHz = plt.axes([0.12, det_slider_bottom-slider_spacing,slider_width,slider_height])
    slider_detGHz = Slider(
        ax_detGHz,
        "Detuning (GHz)",
        det_min_GHz, det_max_GHz,
        valinit=det_init_GHz,
        valfmt='%.3f')
    ui["detGHz_inside_ticks"] = add_slider_ticks_inside_step(ax_detGHz, slider_detGHz, step=1.0)


    slider_detGHz.valtext.set_visible(False)
    
    ax_det_GHz_text = textbox_next_to_slider(ax_detGHz)
    det_text_GHz = TextBox(ax_det_GHz_text,'',initial=f"{det_init_GHz:.3f}",color='0.95', hovercolor='0.95')
    
    # Slew is now DV/ns
    slew_min = 1e-6
    slew_max = 1e1
    
    ax_slew = plt.axes([0.12, det_slider_bottom-slider_spacing*2,slider_width,slider_height])
    slider_slew = Slider(
        ax_slew,
        "Det. rate(DV/ns)",
        slew_min, slew_max,
        valinit=st.detuning_slew_rate,
        valfmt='%.3f'
    )
    ui["slew_inside_ticks"] = add_slider_ticks_inside_step(ax_slew, slider_slew, step=1.0)


    slider_slew.valtext.set_visible(False)
    
    ax_slew_text = textbox_next_to_slider(ax_slew)
    text_slew = TextBox(ax_slew_text, '', initial=f"{st.detuning_slew_rate:.3f}",color='0.95', hovercolor='0.95')
    
    
    
    ax_Pnorm = plt.axes([0.12, det_slider_bottom-slider_spacing*4,slider_width,slider_height])
    Pnorm_slider = Slider(
        ax_Pnorm,
        "Pump Power (norm)",
        Pnorm_min,
        Pnorm_max,
        valinit=st.P_norm,
        valfmt='%.2f'
    )
    ui["Pnorm_inside_ticks"] = add_slider_ticks_inside_step(ax_Pnorm, Pnorm_slider, step=1.0)

    Pnorm_slider.valtext.set_visible(False)
    
    
    ax_Pnorm_text = textbox_next_to_slider(ax_Pnorm)
    Pnorm_text = TextBox(ax_Pnorm_text,'',initial=f"{st.P_norm:.2f}",color='0.95', hovercolor='0.95')
    
    # ---- Physical power slider in mW (derived from Pnorm limits) ----
    P_min_mW  = 1e3 * Pnorm_to_W(Pnorm_min, st.P_in_norm_factor)
    P_max_mW  = 1e3 * Pnorm_to_W(Pnorm_max, st.P_in_norm_factor)
    P_init_mW = 1e3 * Pnorm_to_W(st.P_norm,  st.P_in_norm_factor)
    
    ax_PmW = plt.axes([0.12, det_slider_bottom-slider_spacing*5,slider_width,slider_height])
    PmW_slider = Slider(
        ax_PmW,
        "Pump Power (mW)",
        P_min_mW,
        P_max_mW,
        valinit=P_init_mW,
        valfmt='%.2f'
    )

    PmW_slider.valtext.set_visible(False)
    
    ax_PmW_text = textbox_next_to_slider(ax_PmW)
    PmW_text = TextBox(ax_PmW_text,'',initial=f"{P_init_mW:.2f}",color='0.95', hovercolor='0.95')
    PmW_tick_step = 5.0  # or 10, 20, 50 depending on your range
    ui["PmW_inside_ticks"] = add_slider_ticks_inside_step(
        ax_PmW, PmW_slider,
        step=PmW_tick_step, y0=0.30, y1=0.70, lw=0.7
    )


    syncing = {"flag": False}
    

    def on_Pnorm_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        Pn = float(Pnorm_slider.val)
        st.set_targets(P_norm=Pn)
    
        # sync mW slider
        PmW = 1e3 * Pnorm_to_W(Pn, st.P_in_norm_factor)
        PmW_slider.set_val(PmW)
        Pnorm_text.set_val(f"{Pn:.2f}")
        PmW_text.set_val(f"{PmW:.2f}")
    
        syncing["flag"] = False
        
    def on_Pnorm_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
        
        Pn = float(text)
        
        #passing the input
        st.set_targets(P_norm=Pn)
        
        #updating sliders/boxes
        Pnorm_slider.set_val(Pn)
        PmW = 1e3 * Pnorm_to_W(Pn, st.P_in_norm_factor)
        PmW_slider.set_val(PmW)
        PmW_text.set_val(f"{PmW:.2f}")
        
        
        syncing["flag"] = False

    def on_PmW_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        PmW = float(PmW_slider.val)
        P_W = PmW * 1e-3
        Pn = W_to_Pnorm(P_W, st.P_in_norm_factor)
    

        st.set_targets(P_norm=Pn)
        
        Pnorm_slider.set_val(Pn)
        Pnorm_text.set_val(f"{Pn:.2f}")
        PmW_text.set_val(f"{PmW:.2f}")
    
        syncing["flag"] = False
        
    def on_PmW_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
        
        PmW = float(text)
        P_W = PmW * 1e-3
        Pn = W_to_Pnorm(P_W, st.P_in_norm_factor)
        
        #passing the input
        st.set_targets(P_norm=Pn)
        
        #updating sliders/boxes
        Pnorm_slider.set_val(Pn)
        Pnorm_text.set_val(f"{Pn:.2f}")
        PmW_slider.set_val(PmW)
        
        
        syncing["flag"] = False

    Pnorm_slider.on_changed(on_Pnorm_changed)
    Pnorm_text.on_submit(on_Pnorm_text_submit)
    PmW_slider.on_changed(on_PmW_changed)
    PmW_text.on_submit(on_PmW_text_submit)

    
    def on_det_DV_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
        
        DV = float(det_slider.val)

    
        # drive the target continuously (smooth via st.detuning_slew in step())
        st.set_targets(detuning=DV)
    
        # keep the GHz slider synced
        det_GHz = DV_to_detuning_GHz(DV, st.kappa_avg)
        slider_detGHz.set_val(det_GHz)
        
        #updating textboxes
        det_text_DV.set_val(f"{DV:.2f}")
        det_text_GHz.set_val(f"{det_GHz:.2f}")
    
        syncing["flag"] = False
        
    
    
    def on_det_DV_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
        
        DV = float(text)
        
        #passing the input
        st.set_targets(detuning=DV)
        
        #updating sliders/boxes
        det_slider.set_val(DV)
        det_GHz = DV_to_detuning_GHz(DV, st.kappa_avg)
        slider_detGHz.set_val(det_GHz)
        det_text_GHz.set_val(f"{det_GHz:.2f}")
        
        syncing["flag"] = False
        
        
             
    def on_detGHz_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        det_GHz = float(slider_detGHz.val)
        DV = detuning_GHz_to_DV(det_GHz, st.kappa_avg)
        
        
        #other sliders
        det_text_DV.set_val(f"{DV:.2f}")
        det_text_GHz.set_val(f"{det_GHz:.2f}")
        det_slider.set_val(DV)
        
        
        st.set_targets(detuning=DV)
    
        syncing["flag"] = False
   
    
    def on_det_GHz_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
        
        det_GHz = float(text)
        
        #passing the input
        DV = detuning_GHz_to_DV(det_GHz, st.kappa_avg)
        st.set_targets(detuning=DV)
        
        #updating sliders/boxes
        det_slider.set_val(DV)
        slider_detGHz.set_val(det_GHz)
        det_text_DV.set_val(f"{DV:.2f}")
        
        
        syncing["flag"] = False
        
    det_slider.on_changed(on_det_DV_changed)
    det_text_DV.on_submit(on_det_DV_text_submit)
    slider_detGHz.on_changed(on_detGHz_changed)    
    det_text_GHz.on_submit(on_det_GHz_text_submit)
    
    def slew_DVns_to_GHzns(slew_DVns, kappa_avg):
        # detuning_GHz = DV * kappa/(2π) * 1e-9
        # so derivative: (GHz/ns) = (DV/ns) * kappa/(2π) * 1e-9
        return slew_DVns * kappa_avg / (2*np.pi) * 1e-9
    
    def slew_GHzns_to_DVns(slew_GHzns, kappa_avg):
        return slew_GHzns * 1e9 * (2*np.pi) / kappa_avg

    
    def on_slew_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        slew_DVns = float(slider_slew.val)
        st.detuning_slew_rate = slew_DVns
        text_slew.set_val(f"{slew_DVns:.2f}")
    
        # --- NEW: sync GHz/ns slider + textbox ---
        slew_GHzns = slew_DVns_to_GHzns(slew_DVns, st.kappa_avg)
        slider_slewGHz.set_val(slew_GHzns)
        text_slewGHz.set_val(f"{slew_GHzns:.3f}")
    
        syncing["flag"] = False

        
    def on_slewGHz_changed(_val):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        slew_GHzns = float(slider_slewGHz.val)
        slew_DVns = slew_GHzns_to_DVns(slew_GHzns, st.kappa_avg)
    
        st.detuning_slew_rate = slew_DVns
    
        # sync DV/ns slider + both textboxes
        slider_slew.set_val(slew_DVns)
        text_slew.set_val(f"{slew_DVns:.2f}")
        text_slewGHz.set_val(f"{slew_GHzns:.3f}")
    
        syncing["flag"] = False
    
    
    def on_slewGHz_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        slew_GHzns = float(text)
        slew_DVns = slew_GHzns_to_DVns(slew_GHzns, st.kappa_avg)
    
        st.detuning_slew_rate = slew_DVns
    
        slider_slew.set_val(slew_DVns)
        slider_slewGHz.set_val(slew_GHzns)
        text_slew.set_val(f"{slew_DVns:.2f}")
    
        syncing["flag"] = False

    def on_slew_text_submit(text):
        if syncing["flag"]:
            return
        syncing["flag"] = True
    
        slew = float(text)
        st.detuning_slew_rate = slew
        slider_slew.set_val(slew)
    
        syncing["flag"] = False

    slider_slew.on_changed(on_slew_changed)
    text_slew.on_submit(on_slew_text_submit)
    
    # ---- Physical slew slider in GHz/ns (derived from DV/ns limits) ----
    slew_min_GHzns = slew_DVns_to_GHzns(slew_min, st.kappa_avg)
    slew_max_GHzns = slew_DVns_to_GHzns(slew_max, st.kappa_avg)
    slew_init_GHzns = slew_DVns_to_GHzns(st.detuning_slew_rate, st.kappa_avg)
    
    ax_slewGHz = plt.axes([0.12, det_slider_bottom - slider_spacing*3,slider_width,slider_height])
    slider_slewGHz = Slider(
        ax_slewGHz,
        "Det. rate (GHz/ns)",
        slew_min_GHzns, slew_max_GHzns,
        valinit=slew_init_GHzns,
        valfmt="%.3f"
    )
    # --- Inside ticks for slew in GHz/ns ---
    slewGHz_tick_step = 0.5  # pick what "one tick" means for you (0.1, 0.5, 1.0 ...)
    ui["slewGHz_inside_ticks"] = add_slider_ticks_inside_step(
        ax_slewGHz, slider_slewGHz,
        step=slewGHz_tick_step, y0=0.30, y1=0.70, lw=0.7
    )

    slider_slewGHz.valtext.set_visible(False)
    
    ax_slewGHz_text = textbox_next_to_slider(ax_slewGHz)
    text_slewGHz = TextBox(ax_slewGHz_text, "", initial=f"{slew_init_GHzns:.3f}",color='0.95', hovercolor='0.95')
    
    slider_slewGHz.on_changed(on_slewGHz_changed)
    text_slewGHz.on_submit(on_slewGHz_text_submit)
    
    # --- UI SETTINGS PANEL (single source of truth) ---
    SIDEBAR = dict(
        x=0.92,
        w=0.065,
        h=0.04,     # textbox height (increase to avoid squishing)
        top=0.9,
        row=0.07,   # vertical step between rows (increase to avoid label overlap)
        title_fs=10,
        title_pad=2,
        face='0.95',
    
        # buttons
        btn_h=0.04,
        gap_after_fields=0.006,
        gap_after_apply=0.005,
    )
    
    def sidebar_y(idx):
        return SIDEBAR["top"] - idx * SIDEBAR["row"]
    
    def add_sidebar_textbox(key, label_text, initial_val, idx):
        y = sidebar_y(idx)
        ax = plt.axes([SIDEBAR["x"], y, SIDEBAR["w"], SIDEBAR["h"]])
        ax.set_title(label_text, fontsize=SIDEBAR["title_fs"], pad=SIDEBAR["title_pad"])
        tb = TextBox(ax, "", initial=initial_val, color=SIDEBAR["face"], hovercolor=SIDEBAR["face"])
        ui[f"ax_{key}"] = ax
        ui[f"txt_{key}"] = tb
        return ax, tb
    
    def add_sidebar_button(key, label, y):
        ax = plt.axes([SIDEBAR["x"], y, SIDEBAR["w"], SIDEBAR["btn_h"]])
        btn = Button(ax, label)
        ui[f"ax_{key}"] = ax
        ui[f"btn_{key}"] = btn
        return ax, btn

    
    # ---- Add parameters (ONLY ONCE) ----
    fields = [
        ("modes", "Number\nof modes", f"{st.number_modes}"),
        ("fsr",   "FSR [Hz]",         f"{st.fsr:.2e}"),
        ("lambda","Pump λ [nm]",      f"{st.wvl_pump*1e9:.1f}"),
        ("q",     "Q factor",         f"{st.Q:.2e}"),
        ("aeff",  "Aeff [um2]",       f"{st.Aeff*1e12:.2f}"),
        ("n2",    "n2 [m2/W]",        f"{st.n2:.2e}"),
        ("eta",   "Coupling eta",     f"{st.eta:.3f}"),
        ("dint",  "Dispersion\nfile name", st.dint_file_path),
    ]


    
    for i, (key, lab, val) in enumerate(fields):

        y = sidebar_y(i)

        ax = plt.axes([SIDEBAR["x"], y, SIDEBAR["w"], SIDEBAR["h"]])
        ax.set_title(lab, fontsize=SIDEBAR["title_fs"], pad=SIDEBAR["title_pad"])
    
        tb = TextBox(ax, "", initial=val, color=SIDEBAR["face"], hovercolor=SIDEBAR["face"])
        ui[f"ax_{key}"] = ax
        ui[f"txt_{key}"] = tb

    # --- Make sidebar widgets reliably clickable ---
    SIDEBAR_KEYS = ["modes", "fsr", "lambda", "q", "aeff", "n2", "eta", "dint"]
    for k in SIDEBAR_KEYS:
        axk = ui.get(f"ax_{k}")
        tbk = ui.get(f"txt_{k}")
        if axk is not None:
            axk.set_navigate(False)      # don't let toolbar/navigation steal events
            axk.set_zorder(2000)         # bring in front
            axk.patch.set_alpha(1.0)     # ensure it receives clicks
        if tbk is not None:
            tbk.set_active(True)
    
    fig.canvas.draw_idle()

    
    # ---- Apply button (below the last field) ----
    apply_y = sidebar_y(len(fields)) - SIDEBAR["gap_after_fields"]

    ui["ax_apply"] = plt.axes([SIDEBAR["x"], apply_y, SIDEBAR["w"], SIDEBAR["btn_h"]])
    ui["btn_apply"] = Button(ui["ax_apply"], "Apply\n& Reset")
    
    # ---- Uniform vertical spacing for sidebar buttons ----
    BTN_GAP = 0.020   # <-- master spacing between buttons (increase for more air)
    
    # Apply button already defined above as apply_y
    
    y = apply_y - (SIDEBAR["btn_h"] + BTN_GAP)
    
    ui["ax_save_plot"] = plt.axes([SIDEBAR["x"], y, SIDEBAR["w"], SIDEBAR["btn_h"]])
    ui["btn_save_plot"] = Button(ui["ax_save_plot"], "Save plot")
    
    y -= (SIDEBAR["btn_h"] + BTN_GAP)
    ui["ax_save_data"] = plt.axes([SIDEBAR["x"], y, SIDEBAR["w"], SIDEBAR["btn_h"]])
    ui["btn_save_data"] = Button(ui["ax_save_data"], "Save data")
    
    y -= (SIDEBAR["btn_h"] + BTN_GAP)
    ui["ax_save_both"] = plt.axes([SIDEBAR["x"], y, SIDEBAR["w"], SIDEBAR["btn_h"]])
    ui["btn_save_both"] = Button(ui["ax_save_both"], "Save both")
    
    y -= (SIDEBAR["btn_h"] + BTN_GAP)
    ui["ax_run"] = plt.axes([SIDEBAR["x"], y, SIDEBAR["w"], SIDEBAR["btn_h"]])
    ui["btn_run"] = Button(ui["ax_run"], "Run/Pause")
    ui["btn_run"].label.set_text("Run")  # start paused
    fig.canvas.draw_idle()

    drag_state = {"active": False}

    running = {"flag": False}   # or False if you want it to start paused
     
    def on_press(event):
        if event.inaxes == ax_det:
            drag_state["active"] = True
    
    def on_release(event):
        if drag_state["active"]:
            drag_state["active"] = False
            st.set_targets(detuning=det_slider.val)  # commit once on release
            
    anim_ref = {"ani": None}  # defined BEFORE on_run_clicked

    def on_run_clicked(_event):
        running["flag"] = not running["flag"]
        state = "Pause" if running["flag"] else "Run"
        ui["btn_run"].label.set_text(state)
    
        if running["flag"]:
            # Resume: give control back to blitting
            set_anim_artists_animated(True)
            if hasattr(fig, "_blit_cache"):
                fig._blit_cache.clear()
            ani.event_source.start()
            ani._step()  # optional kick
        else:
            # Pause: stop timer AND make artists non-animated so they survive repaints
    
            set_anim_artists_animated(False)
            fig.canvas.draw()   # now the paused frame is a normal static draw
    
            # optional, but fine to keep
            if hasattr(fig, "_blit_cache"):
                fig._blit_cache.clear()
    ANIM_ARTISTS = [stem_lc, stem_sc, baseline_line, pulse_line, line_Pcav_W, line_DV, line_Pnorm, hud]
    
    def set_anim_artists_animated(is_animated: bool):
        for a in ANIM_ARTISTS:
            try:
                a.set_animated(is_animated)
            except Exception:
                pass  # some artist types may not support it

    def save_spectrum_snapshot(_event=None):
        """
        Save a snapshot of:
          1) THROUGH-PORT physical spectrum (dBm + W)  <-- matches GUI plot
          2) Intracavity temporal pulse intensity (normalized)  <-- same as before
        """
        try:
            # ---------------------------
            # 1) Through-port spectrum
            # ---------------------------
            # This is exactly what your GUI plots
            spec_through_dBm = st.get_output_spectrum_dBm()
    
            # Also compute linear through-port power [W] per mode,
            # using the same physics used inside get_output_spectrum_dBm():
            # s_out = s_in - sqrt(kappa_ex) * a_mu_phys
            # P_out = hbar * omega_mu * |s_out|^2
            A_mu_norm = st.spectrum_E_t_fast_norm / st.number_modes
            a_mu_phys = st.E_amp_2_norm_factor * A_mu_norm
    
            P_in_phys_W = st.P_norm * st.P_in_norm_factor
    
            s_in = np.zeros_like(a_mu_phys, dtype=complex)
            s_in[st.pump_idx] = np.sqrt(max(P_in_phys_W, 0.0) / (st.hbar * st.ome_pump))
    
            s_out = s_in - np.sqrt(st.kappa_ex) * a_mu_phys
            P_through_W = st.hbar * st.ome_grid * (np.abs(s_out) ** 2)
            P_through_W = np.maximum(P_through_W, 1e-30)  # avoid log/zeros
    
            # ---------------------------
            # Output folder
            # ---------------------------
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                base_dir = os.getcwd()
    
            out_dir = os.path.join(base_dir, "exported_data")
            os.makedirs(out_dir, exist_ok=True)
    
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"spectrum_{ts}_j{st.j:09d}_DV{st.DV:+.4f}.csv"
            fpath = os.path.join(out_dir, fname)
    
            data = np.column_stack([
                st.mu.astype(int),
                st.frq_grid,
                st.frq_grid * 1e-12,
                st.wvl_grid,
                st.wvl_grid * 1e9,
                spec_through_dBm,
                P_through_W,
            ])
    
            header = (
                "Snapshot of LLE THROUGH-PORT spectrum\n"
                f"timestamp={ts}, j={st.j}, DV={st.DV}, P_in_W={P_in_phys_W}, kappa_ex={st.kappa_ex}\n"
                "columns: mu, f_Hz, f_THz, lambda_m, lambda_nm, P_through_dBm, P_through_W"
            )
    
            np.savetxt(fpath, data, delimiter=",", header=header, comments="# ")
            print(f"[Snapshot] Saved THROUGH-PORT spectrum to: {fpath}")
    
            # ---------------------------
            # 2) Temporal pulse snapshot (same as before)
            # ---------------------------
            tname = f"pulse_{ts}_j{st.j:09d}_DV{st.DV:+.4f}.csv"
            tpath = os.path.join(out_dir, tname)
    
            I_t = np.abs(st.E_t_fast_norm) ** 2
            tdata = np.column_stack([st.tau_ps, I_t])
    
            theader = (
                "Snapshot of intracavity temporal pulse\n"
                f"timestamp={ts}, j={st.j}, DV={st.DV}, S={st.S}\n"
                "columns: tau_ps, |E(tau)|^2_norm"
            )
    
            np.savetxt(tpath, tdata, delimiter=",", header=theader, comments="# ")
            print(f"[Snapshot] Saved pulse to: {tpath}")
    
            # On success:
            flash_button(ui["btn_save_data"], "Saved!", "Save data")
    
        except Exception as e:
            print("[Snapshot] FAILED:", repr(e))

            

        
    def save_visual_plot():
        """Saves the current GUI dashboard as a high-res PNG."""
        try:
            out_dir = os.path.join(os.getcwd(), "saved_plots")
            os.makedirs(out_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"LLE_Plot_{ts}_N{st.number_modes}.png"
            # bbox_inches='tight' is critical to include the sidebar in the image
            fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
            print(f"Plot saved: {fname}")
            return True
        except Exception as e:
            print(f"Plot save error: {e}")
            return False

    def save_raw_data():
        """Calls your existing snapshot logic to save CSV files."""
        try:
            save_spectrum_snapshot()
            return True
        except Exception as e:
            print(f"Data save error: {e}")
            return False
    
    def on_save_plot_clicked(event):
        if save_visual_plot():
            flash_button(ui['btn_save_plot'], "Saved!", "Save plot")

    def on_save_data_clicked(event):
        if save_raw_data():
            flash_button(ui['btn_save_data'], "Saved!", "Save data")


    def on_save_both_clicked(event):
        plot_ok = save_visual_plot()
        data_ok = save_raw_data()
        if plot_ok and data_ok:
            flash_button(ui['btn_save_both'], "Saved!", "Save both")

    
    def apply_new_params(event):
        # 1. Temporarily pause the simulation
        was_running = running["flag"]
        running["flag"] = False
        
        try:
            # 2. Get new values
            new_n = int(ui['txt_modes'].text)
            st.wvl_pump = float(ui['txt_lambda'].text) * 1e-9
            st.fsr = float(ui['txt_fsr'].text)
            st.Q = float(ui['txt_q'].text)
            st.Aeff = float(ui['txt_aeff'].text) * 1e-12
            st.n2 = float(ui['txt_n2'].text)
            st.eta = float(ui['txt_eta'].text)
            st.dint_file_path = ui['txt_dint'].text
            #st.dint_file_choice = True
            
            # update pump physics
            st.frq_pump = st.c / st.wvl_pump
            st.ome_pump = 2*np.pi * st.frq_pump
            
            # 3. Complete re-gridding in the LLEState
            st.reinitialize(new_n_modes=new_n)
            
            # --- Reset core sim targets to the startup defaults ---
            st.detuning_slew_rate = DEFAULTS["slew"]
            
            st.DV = DEFAULTS["DV"]
            st.target_detuning = st.DV
            
            # st.P_norm = DEFAULTS["Pnorm"]
            # st.P_norm_target = st.P_norm
            # st.S = np.sqrt(st.P_norm)
            # st.S_target = st.S
            
            # # Rebuild CW pump input to match reset S (simple + robust)
            # st.tE_in = np.full(st.mu.shape, st.S)
            # st.E_in = fft(st.tE_in)
            # st.fE_in_o = fftshift(st.E_in)
            # st.fE_in = st.fE_in_o
            # st.input_pump_half_step = st.fE_in * (st.tal_step / 2)

            
            # --- Reset GUI widgets (avoid feedback loops) ---
            syncing["flag"] = True
            
            # Detuning sliders/text
            det_slider.set_val(st.DV)
            det_text_DV.set_val(f"{st.DV:.2f}")
            
            det_GHz = DV_to_detuning_GHz(st.DV, st.kappa_avg)
            slider_detGHz.set_val(det_GHz)
            det_text_GHz.set_val(f"{det_GHz:.2f}")
            
            # Slew sliders/text (DV/ns and GHz/ns)
            slider_slew.set_val(st.detuning_slew_rate)
            text_slew.set_val(f"{st.detuning_slew_rate:.2f}")
            
            slew_GHzns = slew_DVns_to_GHzns(st.detuning_slew_rate, st.kappa_avg)
            slider_slewGHz.set_val(slew_GHzns)
            text_slewGHz.set_val(f"{slew_GHzns:.3f}")
            
            # Power sliders/text (norm and mW)
            Pnorm_slider.set_val(st.P_norm)
            Pnorm_text.set_val(f"{st.P_norm:.2f}")
            
            PmW = 1e3 * Pnorm_to_W(st.P_norm, st.P_in_norm_factor)
            PmW_slider.set_val(PmW)
            PmW_text.set_val(f"{PmW:.2f}")
            # Fewer ticks for mW (choose a coarser step)
            PmW_tick_step = 5.0   # try 5 mW; if still too dense, use 10, 20, 50...
            ui["PmW_inside_ticks"] = add_slider_ticks_inside_step(ax_PmW, PmW_slider, step=PmW_tick_step, y0=0.30, y1=0.70, lw=0.7)

            
            syncing["flag"] = False

            
            # 4. CRITICAL: Update global variables used in update()
            # 4. CRITICAL: recompute axes from new N and FSR
            nonlocal x_THz, t_ps
            x_THz, t_ps = sync_x_axes_to_grid(st, ax_spec, ax_pulse, ax_spec_top, pad_frac=0.01)
            
            # 5. Update plot line/artist data shapes
            spec_line.set_data(x_THz, np.full(new_n, -100))
            
            stem_sc.set_offsets(np.column_stack([x_THz, np.full(new_n, -300)]))
            
            pulse_line.set_data(t_ps, np.zeros(new_n))

            
            # Clear old stem segments so spectrum truly resets visually
            stem_lc.set_segments([])
            
            # Put baseline line where you expect after reset (optional)
            baseline_state["y"] = SPECTRUM_YMIN  # or -200, etc.
            baseline_line.set_ydata([baseline_state["y"], baseline_state["y"]])
            
            # Force a full redraw (blit-safe)
            fig.canvas.draw()

            # If FuncAnimation blit cache exists, clear it (prevents stale background)
            if hasattr(fig, "_blit_cache"):
                fig._blit_cache.clear()


            # 6. Reset history buffers
            nonlocal pow_x, y_Pcav_W, y_DV, y_Pnorm
            pow_x = np.full(Nkeep, np.nan)
            y_Pcav_W = np.full(Nkeep, np.nan)
            y_DV = np.full(Nkeep, np.nan)
            y_Pnorm = np.full(Nkeep, np.nan)
            pow_i["i"] = 0
            
            fig.canvas.draw_idle()
            print(f"Simulation restarted with {new_n} modes.")
            
        except Exception as e:
            print(f"Update error: {e}")
        
        # 8. Restore previous run state
        running["flag"] = was_running
        
    def flash_button(btn, temp_text, final_text, delay_ms=500):
        """
        Temporarily change button label, then restore after delay.
        Non-blocking (does not freeze animation).
        """
        btn.label.set_text(temp_text)
        fig.canvas.draw_idle()
    
        timer = fig.canvas.new_timer(interval=delay_ms)
    
        def restore():
            btn.label.set_text(final_text)
            fig.canvas.draw_idle()
    
        timer.add_callback(restore)
        timer.start()

        
    # ---- Hook up sidebar buttons ----
    
    ui["btn_apply"].on_clicked(apply_new_params)

    
    ui["btn_save_plot"].on_clicked(on_save_plot_clicked)
    ui["btn_save_data"].on_clicked(on_save_data_clicked)
    ui["btn_save_both"].on_clicked(on_save_both_clicked)
    
    ui["btn_run"].on_clicked(on_run_clicked)

    
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    
    # ---- Animation update ----
    def update(_frame):
        if not running["flag"]:
                return stem_lc, stem_sc, baseline_line, pulse_line, line_Pcav_W, line_DV, line_Pnorm, hud

    
        # 1) advance simulation
        st.step(n_steps=steps_per_frame)
        
        # ---- Physical readouts ----
        det_GHz   = st.DV * st.kappa_avg / (2*np.pi) * 1e-9           # GHz
        Pin_mW    = 1e3 * (st.P_norm * st.P_in_norm_factor)            # mW

        
        # ---- Update temporal pulse ----
        I_t = np.abs(st.E_t_fast_norm)**2
        
        if st.temporal_interpol:
            print("interpolating")
            ##interpolating the temporal fast time to make temporal pulse profile graph prittier
            interp_func = interp1d(t_ps, I_t, kind='cubic')
            t_fine = np.linspace(t_ps[0], t_ps[-1], len(t_ps) * 10)
            t_pulse = t_fine
            I_t_fine = interp_func(t_fine)
            I_t_pulse = I_t_fine
        else:
            t_pulse = t_ps
            I_t_pulse = I_t
        pulse_line.set_xdata(t_pulse)
        pulse_line.set_ydata(I_t_pulse)
    
        # 2) spectrum (OSA-like, wavelength axis)
        y_db = st.get_output_spectrum_dBm()
        
        b = baseline_state["y"]
        baseline_line.set_ydata([b, b])
        
        # Keep “floor at baseline” style if you want
        y_plot = np.maximum(y_db, b)
        
        mask = y_db > b
        segments = [((x, b), (x, y)) for x, y in zip(x_THz[mask], y_plot[mask])]
        stem_lc.set_segments(segments)
        
        # dots for all modes
        stem_sc.set_offsets(np.column_stack([x_THz, y_plot]))
        spec_line.set_data(x_THz, y_db)



        # Keep main title static; update only the subtitle text
        t_phys_ns = 1e9 * (2.0 / st.kappa_avg) * (st.j * st.tal_step)
        
        # ---- Power plot (3 traces ONLY) ----
        #j_now = float(st.j)
        
        # Physical slow time [µs]
        t_phys_ns = 1e9 * (2.0 / st.kappa_avg) * (st.j * st.tal_step)
        

        # Pcav_norm = float(st.get_intracavity_mean())   # ⟨|E|²⟩ directly from field
        # Pcav_mW   = Pcav_norm_to_mW(Pcav_norm)         # consistent physical conversion
        
        # --- Physical intracavity circulating power (from modal amplitudes) ---
        
        # Normalized mode amplitudes (consistent with get_output_spectrum_dBm())
        A_mu_norm = st.spectrum_E_t_fast_norm / st.number_modes
        
        # Convert to physical intracavity mode amplitudes
        a_mu_phys = st.E_amp_2_norm_factor * A_mu_norm
        
        # Intracavity energy stored across all modes [J]
        U_J = np.sum(st.hbar * st.ome_grid * (np.abs(a_mu_phys) ** 2))
        
        # Circulating power [W] = energy / round-trip time
        Pcav_W = U_J / st.t_phys_round_trip
        Pcav_mW = 1e3 * Pcav_W
        
        # Normalized mean intensity (your original meaning)
        Pcav_norm = float(np.mean(np.abs(st.E_t_fast_norm) ** 2))


        
        DV_now    = float(st.DV)
        Pnorm_now = float(st.P_norm)
        
        slew_GHzns = slew_DVns_to_GHzns(st.detuning_slew_rate, st.kappa_avg)
        
        hud.set_text(
            f"Simulation time (t_sim): {st.j * st.tal_step:7.2f} s"
            f"Physical time (t_phys): {t_phys_ns:7.2f} ns\n"
            f"Detuning (DV): {DV_now:6.2f} norm / {DV_to_detuning_GHz(DV_now, st.kappa_avg):7.2f} GHz\n"
            f"Det. rate (DV_r): {st.detuning_slew_rate:6.2f} DV/ns / {slew_GHzns:7.2f} GHz/ns\n"
            f"Pump power (P_in): {Pnorm_now:6.2f} norm / {Pin_mW:7.2f} mW\n"
            f"Intracavity power (P_cav): {Pcav_norm:6.2f} norm / {Pcav_mW:7.2f} mW"
        )

        k = pow_i["i"] % Nkeep
        #pow_x[k]    = j_now
        pow_x[k] = t_phys_ns
        y_Pcav_W[k] = Pcav_mW
        y_DV[k]     = DV_now
        y_Pnorm[k]  = Pnorm_now
        pow_i["i"] += 1

        # unwrap ring buffer
        if pow_i["i"] < Nkeep:
            x_plot    = pow_x[:pow_i["i"]]
            Pcav_plot = y_Pcav_W[:pow_i["i"]]
            DV_plot   = y_DV[:pow_i["i"]]
            Pn_plot   = y_Pnorm[:pow_i["i"]]
        else:
            order     = np.arange(k + 1, k + 1 + Nkeep) % Nkeep
            x_plot    = pow_x[order]
            Pcav_plot = y_Pcav_W[order]
            DV_plot   = y_DV[order]
            Pn_plot   = y_Pnorm[order]

        # set data
        line_Pcav_W.set_data(x_plot, Pcav_plot)
        line_DV.set_data(x_plot, DV_plot)
        line_Pnorm.set_data(x_plot, Pn_plot)

        WINDOW_NS = 1000.0  # show last 20 ns
        
        if np.isfinite(x_plot).any():
            xmax = float(np.nanmax(x_plot))
            ax_pow.set_xlim(max(0.0, xmax - WINDOW_NS), max(WINDOW_NS, xmax))


        return stem_lc, stem_sc, baseline_line, pulse_line, line_Pcav_W, line_DV, line_Pnorm, hud

    ani = animation.FuncAnimation(fig, update, interval=interval_ms, blit=True, cache_frame_data=False)

    fig._ani = ani  # keep alive in Spyder
    fig._ui = ui 
    
    anim_ref["ani"] = ani
    
    ani.event_source.stop()  # <-- start paused for real (not just via the flag)

    global APP
    APP = PycombsApp()
    APP.fig = fig
    APP.ui = ui
    APP.st = st
    APP.ani = ani


    plt.show()



if __name__ == "__main__":
    run_gui()


    # # SSFM solving loop
    # for j in range(1, N_iter + 1):
    #     if j % save_step_point == 0:
    #         E_t_fast_norm = ifft(fftshift(spectrum_E_t_fast_norm)) # Takes the spectrum of the normalized E field intracavity and obtains the E field in the time domain
    #         SaveE_t_fast_norm[j // save_step_point - 1, :] = E_t_fast_norm # Stores this time-domain field snapshot into a 2D array
    #         SaveDet[j // save_step_point - 1] = DV # Saves the current detuning value DV corresponding to this snapshot
    #         noise = np.random.normal(0, pump_noise_amp, len(mu)) + 1j * np.random.normal(0, pump_noise_amp, len(mu)) # Creates fresh vacuum/shot-noise fluctuations for the pump channel.
    #         fE_in = fE_in_o + noise # Adds noise onto the pump spectrum
    #         input_pump_half_step = fE_in * (tal_step / 2) # Convert pump into the half-step injection term

    #     DV += delta_detuning_norm_int_step # Increases the Detuning Value (DV) by one normalized detuning step unit
        
    #     # Application of the first half step of the linear operator
    #     lin_oper = -(kappa_all / kappa_avg) - 1j * DV - 1j * dint_norm #: Linear evolution operator of the LLE, combining cavity loss, laser–cavity detuning, and dispersion for each mode in normalized units.
    #     spectrum_E_t_fast_norm = np.exp(lin_oper * (tal_step / 2)) * (spectrum_E_t_fast_norm + input_pump_half_step) # First half step of the linear operator

    #     # Return of the E field to the time domain and application of the full nonlinear step
    #     E_t_fast_norm = ifft(fftshift(spectrum_E_t_fast_norm)) # Takes the spectrum back to the time domain
    #     E_t_fast_norm = np.exp(1j * np.abs(E_t_fast_norm)**2 * tal_step) * E_t_fast_norm

    #     # Return to of the E field to the frequency domain and application of the second half linear step
    #     spectrum_E_t_fast_norm = fftshift(fft(E_t_fast_norm))
    #     spectrum_E_t_fast_norm = np.exp(lin_oper * (tal_step / 2)) * (spectrum_E_t_fast_norm + input_pump_half_step) # Second half step of the linear operator
    #     spectrum_E_t_fast_norm_now = fftshift(fft(E_t_fast_norm)) / number_modes
        
    #     t_phys_now = (2 / kappa_avg) * (j * tal_step) # Calculating physical time from normalized time\
        
    #     # Calculation of amplitudes in the normalized (A) and physical (a) points of view
    #     A_mu_norm = spectrum_E_t_fast_norm_now #[norm] #: Normalized mode amplitudes
    #     a_mu_phys = E_amp_2_norm_factor * A_mu_norm #[zzd] #: Physical mode amplitudes

    #     # Calculation of intracavity power per mode and total in the normalized and physical points of view
    #     P_mu_norm = np.abs(A_mu_norm)**2 #[norm] #: Normalized power stored per mode
    #     P_tot_norm = np.sum(P_mu_norm) #[norm] #: Total normalized power stored per mode
    #     P_mu_phys = hbar * ome_grid * np.abs(a_mu_phys)**2 #[zzd W] #: Physical power stored per mode
    #     P_tot_phys = np.sum(P_mu_phys) #[zzd W] #: Total physical power stored per mode
        
    #     ## Calculation of optical power per mode and total coupled out of the resonator by the bus waveguide

    #     # Input field of the pump
    #     s_in = np.zeros_like(a_mu_phys, dtype=complex)
    #     pump_idx = np.where(mu == 0)[0][0]
    #     s_in[pump_idx] = np.sqrt(P_in_phys / (hbar * ome_pump))
        
    #     s_out = s_in - np.sqrt(kappa_ex) * a_mu_phys #[zzd] #: Output field on the waveguide
        
    #     P_mu_out_phys = hbar * ome_grid * np.abs(s_out)**2 #[W] #: Output power per mode
        
    #     P_mu_out_phys_dBm = 10*np.log10(P_mu_out_phys / 1e-3) #[zzd dbm] #: Output power per mode in dbm
        
 


    


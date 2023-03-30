import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# PROGRAM TO DERIVE MOMENT-CURVATURE RELATIONS FOR RC MEMBERS

# Title of the page
st.title('Moment-curvature relations for RC members', anchor=None)

# Subtitle refering to types of parameters
st.write('Display type of parameters:')
# 3 types of tabs are displayed
tab1, tab2, tab3 = st.tabs(["Geometry", "Material", "Others"])

# Headline of each tab
with tab1:
   st.header("Geometric parameters")
with tab2:
   st.header("Material parameters")
with tab3:
   st.header("Other parameters")

# An itinial dictionary is created to get all keys of the parameters
# List of keys
keyList_parameters = ['Label', 'Symbol', 'Value', 'Status']
geo_parameters = {key: None for key in keyList_parameters}
mat_parameters = {key: None for key in keyList_parameters}
oth_parameters = {key: None for key in keyList_parameters}

# if 'geo' not in st.session_state:
#     st.session_state.geo = geo_parameters
# if 'mat' not in st.session_state:
#     st.session_state.mat = mat_parameters
# if 'oth' not in st.session_state:
#     st.session_state.oth = oth_parameters

# Callback functions to collect the save parameters
# def geo_collector(**geo_dic):
#     geo_parameters = geo_dic
#     return geo_parameters

# def mat_collector(**mat_dic):
#     mat_parameters = mat_dic
#     return mat_parameters

# def oth_collector(**oth_dic):
#     oth_parameters = oth_dic
#     return oth_parameters

def flatten(l):
    return [item for sublist in l for item in sublist]

# GEOMETRY PARAMETERS
with tab1:

    label = 'Width'
    b = st.number_input(label, min_value=0.0, format='%.2f')  # GUI to width
    # st.write('b:', b)
    if b != 0:
        check_parameter = True
    else:
        check_parameter = False
    geo_parameters.update({'Label':[label],'Symbol':['b'],'Value':[b],'Status':[check_parameter]})

    label = 'Depth'
    h = st.number_input(label, min_value=0.0, format='%.2f')  # GUI to depth
    # st.write('h:', h)
    if h != 0:
        check_parameter = True
    else:
        check_parameter = False
    geo_parameters['Label'].append(label)
    geo_parameters['Symbol'].append('h')
    geo_parameters['Value'].append(h)
    geo_parameters['Status'].append(check_parameter)

    label = 'Distance from bottom face to reinforcement in tension'
    x1 = st.number_input(label, min_value=0.0, format='%.2f')   # GUI to distance from bottom face to reinforcement in tension
    # st.write('x1:', x1)
    if x1 != 0:
        check_parameter = True
    else:
        check_parameter = False
    geo_parameters['Label'].append(label)
    geo_parameters['Symbol'].append('x1')
    geo_parameters['Value'].append(x1)
    geo_parameters['Status'].append(check_parameter)

    label = 'Distance from bottom face to reinforcement in compression'
    x2 = st.number_input(label, min_value=0.0, format='%.2f')   # GUI to distance from bottom face to reinforcement in compression
    # st.write('x2:', x2)
    if x2 != 0:
        check_parameter = True
    else:
        check_parameter = False
    geo_parameters['Label'].append(label)
    geo_parameters['Symbol'].append('x2')
    geo_parameters['Value'].append(x2)
    geo_parameters['Status'].append(check_parameter)

    label = 'Area of reinforcement in tension'
    As1 = st.number_input(label, min_value=0.0, format='%.4f')  # GUI to area of reinforcement in tension
    # st.write('As1:', As1)
    if As1 != 0:
        check_parameter = True
    else:
        check_parameter = False
    geo_parameters['Label'].append(label)
    geo_parameters['Symbol'].append('As1')
    geo_parameters['Value'].append(As1)
    geo_parameters['Status'].append(check_parameter)

    label = 'Area of reinforcement in compression'
    As2 = st.number_input(label, min_value=0.0, format='%.4f')  # GUI to area of reinforcement in compression
    # st.write('As2:', As2)
    geo_parameters['Label'].append(label)
    geo_parameters['Symbol'].append('As2')
    geo_parameters['Value'].append(As2)
    geo_parameters['Status'].append(True)

    # I should check if this step is really necessary
    # Saving geometric parameters
    # action_geo = st.button('Save parameters', key=1, on_click=geo_collector, kwargs=st.session_state.geo)


# b = 0.20        # width
# h = 0.50        # depth
# x1 = 0.05       # distance from bottom face to reinforcement in tension
# x2 = 0.45       # distance from bottom face to reinforcement in compression
# As1 = 0.0060    # area of reinforcement in tension
# As2 = 0.0000    # area of reinforcement in compression

#MATERIAL PARAMETERS
with tab2:

    label = 'Concrete compressive strength'
    fc = st.number_input(label, format='%.2f') # GUI to concrete compressive strength
    # st.write('fc:', fc)
    if fc != 0:
        check_parameter = True
    else:
        check_parameter = False
    mat_parameters.update({'Label':[label],'Symbol':['fc'],'Value':[fc],'Status':[check_parameter]})  

    label = 'Concrete tensile strength'
    fct = st.number_input(label, format='%.2f') # GUI to concrete tensile strength
    # st.write('fct:', fct)
    mat_parameters['Label'].append(label)
    mat_parameters['Symbol'].append('fct')
    mat_parameters['Value'].append(fct)
    mat_parameters['Status'].append(True)

    label = 'Concrete strain to reach peak of stress'
    ec2 = st.number_input(label, format='%.4f') # GUI to concrete strain to reach peak of stress
    # st.write('ec2:', ec2)
    if ec2 != 0:
        check_parameter = True
    else:
        check_parameter = False
    mat_parameters['Label'].append(label)
    mat_parameters['Symbol'].append('ec2')
    mat_parameters['Value'].append(ec2)
    mat_parameters['Status'].append(check_parameter)

    label = 'Concrete ultimate strain'
    ecu = st.number_input('Concrete ultimate strain', format='%.4f') # GUI to concrete ultimate strain
    # st.write('ecu', ecu)
    if ecu != 0:
        check_parameter = True
    else:
        check_parameter = False
    mat_parameters['Label'].append(label)
    mat_parameters['Symbol'].append('ecu')
    mat_parameters['Value'].append(ecu)
    mat_parameters['Status'].append(check_parameter)

    label = 'Modulus of elasticity of concrete in tension'
    Eci = st.number_input(label, format='%.2f') # GUI to modulus of elasticity of concrete in tension
    # st.write('Eci:', Eci)
    if Eci != 0:
        check_parameter = True
    else:
        check_parameter = False
    mat_parameters['Label'].append(label)
    mat_parameters['Symbol'].append('Eci')
    mat_parameters['Value'].append(Eci)
    mat_parameters['Status'].append(check_parameter)

    label = 'Yield strength of reinforcing steel'
    fy = st.number_input(label, format='%.2f') # GUI to yield strength of reinforcing steel
    # st.write('fy:', fy)
    if fy != 0:
        check_parameter = True
    else:
        check_parameter = False
    mat_parameters['Label'].append(label)
    mat_parameters['Symbol'].append('fy')
    mat_parameters['Value'].append(fy)
    mat_parameters['Status'].append(check_parameter)

    label = 'Modulus of elasticity of reinforcing steel'
    Es = st.number_input(label, format='%.2f') # GUI to modulus of elasticity of reinforcing steel
    # st.write('Es:', Es)
    if Es != 0:
        check_parameter = True
    else:
        check_parameter = False
    mat_parameters['Label'].append(label)
    mat_parameters['Symbol'].append('Es')
    mat_parameters['Value'].append(Es)
    mat_parameters['Status'].append(check_parameter)

    label = 'Limit strain of reinforcing steel'
    esu = st.number_input(label, format='%.2f') # GUI to limit strain of reinforcing steel
    # st.write('esu:', esu)
    if esu != 0:
        check_parameter = True
    else:
        check_parameter = False
    mat_parameters['Label'].append(label)
    mat_parameters['Symbol'].append('esu')
    mat_parameters['Value'].append(esu)
    mat_parameters['Status'].append(check_parameter)

    # Saving geometric parameters
    # action_mat = st.button('Save parameters', key=2, on_click=mat_collector, kwargs=st.session_state.mat)

# # fc = -0.85*40000    # concrete compressive strength
# # fct = 0             # concrete tensile strength
# # ec2 = -0.0020       # concrete strain to reach peak of stress
# # ecu = -0.0035       # concrete ultimate strain
# # Eci = 26000000      # modulus of elasticity of concrete in tension (when needed)  
# # fy = 500000         # yield strength of reinforcing steel
# # Es = 200000000      # modulus of elasticity of reinforcing steel
# # esu = 0.01          # limit strain of reinforcing steel

# OTHER INPUT PARAMETERS
with tab3:

    label = 'Axial load'
    N0 = st.number_input(label + ' tension; - compression', format='%.2f') # GUI to axial load
    # st.write('N0:', N0)
    oth_parameters.update({'Label':[label],'Symbol':['N0'],'Value':[N0],'Status':[True]})

    label = 'Number of division for integration of stresses'
    ndiv = int(st.number_input(label, format='%.2f')) # GUI to number of division for integration of stresses
    # st.write('ndiv:', ndiv)
    if ndiv!= 0:
        check_parameter = True
    else:
        check_parameter = False
    oth_parameters['Label'].append(label)
    oth_parameters['Symbol'].append('ndiv')
    oth_parameters['Value'].append(ndiv)
    oth_parameters['Status'].append(check_parameter)

    label = 'Maximum number of interactions'
    max_iter = int(st.number_input(label, format='%.2f')) # GUI to maximum number of interactions
    # st.write('max_inter:', max_iter)
    if max_iter != 0:
        check_parameter = True
    else:
        check_parameter = False
    oth_parameters['Label'].append(label)
    oth_parameters['Symbol'].append('max_iter')
    oth_parameters['Value'].append(max_iter)
    oth_parameters['Status'].append(check_parameter)

    label = 'Tolerance for convergence of iterative procedures'
    epsilon = st.number_input(label, format='%.3f') # GUI to tolerance for convergence of iterative procedures
    # st.write('epsilon:', epsilon)
    if epsilon != 0:
        check_parameter = True
    else:
        check_parameter = False
    oth_parameters['Label'].append(label)
    oth_parameters['Symbol'].append('epsilon')
    oth_parameters['Value'].append(epsilon)
    oth_parameters['Status'].append(check_parameter)

    # Saving geometric parameters
    # action_oth = st.button('Save parameters', key=3, on_click=oth_collector, kwargs=st.session_state.oth)

# Create a data frame to visualize the saved parameters
# New dictionary
data_dic1 = {'Variable':flatten([geo_parameters['Symbol'],mat_parameters['Symbol'],oth_parameters['Symbol']]),'Value':flatten([geo_parameters['Value'],mat_parameters['Value'],oth_parameters['Value']]),'Label':flatten([geo_parameters['Label'],mat_parameters['Label'],oth_parameters['Label']]),'Status':flatten([geo_parameters['Status'],mat_parameters['Status'],oth_parameters['Status']])}
data_dic2 = {'Variable':flatten([geo_parameters['Symbol'],mat_parameters['Symbol'],oth_parameters['Symbol']]),'Value':flatten([geo_parameters['Value'],mat_parameters['Value'],oth_parameters['Value']])}
# Data frame from new dectionary
df_dic = pd.DataFrame(data_dic2)
print(df_dic)
st.sidebar.dataframe(df_dic)

print(type(h))

# #OTHER INPUT PARAMETERS
# N0 = 0              # axial load (+ tension; - compression)
# ndiv = 100          # number of division for integration of stresses
# max_iter = 1000     # maximum number of interactions
# epsilon = 10**(-3)  # tolerance for convergence of iterative procedures

p_label = np.array(data_dic1['Label'])
p_value = np.array(data_dic1['Value'])
p_status = np.array(data_dic1['Status'])

action_McurvRC = st.button('Run analysis')
if action_McurvRC:
    if all(p_status) == False:
        false_status_indexes = [i for i in range(len(p_status)) if p_status[i] == False]
        labels_adjustment = p_label[false_status_indexes[0]]
        if labels_adjustment != p_label[false_status_indexes[-1]]:
            for j in range(len(false_status_indexes)-1):
                labels_adjustment = labels_adjustment + ', ' + p_label[false_status_indexes[j+1]]
                print(labels_adjustment)
        labels_adjustment = labels_adjustment + '.'               
        st.write('Parameters not properly defined:',labels_adjustment)           
    else:
        dx = h/ndiv         #size of increment for increment
        dcurv = (0.0135/h)/100  #increment of curvature

        #STRESS-STRAIN RELATION FOR CONCRETE
        def sigma_c(ec):
            if (ec <= fct / Eci) and (ec >= 0):
                sc = ec*Eci
            elif (ec < 0) and (ec >= ec2):
                sc = fc * (1 - (1 - ec / ec2) ** 2)
            elif (ec < ec2):
                sc = fc
            else:
                sc = 0
            return sc

        #STRESS STRAIN RELATION FOR REINFORCING STEEL
        def sigma_s(es):
            if (es <= fy/Es) and (es >= -fy/Es):
                ss = es*Es
            elif (es < -fy/Es):
                ss = -fy
            else:
                ss = fy
            return ss

        #BALANCE OF FORCES
        def somaF(xln,curv):
            F = sigma_s(curv * (xln - x1)) * As1 + sigma_s(curv * (xln - x2)) * As2 - N0
            for j in range(0, ndiv + 1):
                xj = j * dx
                restoy = divmod(j, 2)
                if (j == 0) or (j == ndiv):
                    cjy = 1
                elif restoy[1] == 0:
                    cjy = 2
                else:
                    cjy = 4
                F = F + cjy * dx/3 * (sigma_c(curv*(xln-xj))*b)
            return F

        #BALANCE OF MOMENTS
        def somaM(xln,curv):
            M = -sigma_s(curv * (xln - x1)) * As1*x1 - sigma_s(curv * (xln - x2)) * As2*x2 + N0*h/2
            for j in range(0, ndiv + 1):
                xj = j * dx
                restoy = divmod(j, 2)
                if (j == 0) or (j == ndiv):
                    cjy = 1
                elif restoy[1] == 0:
                    cjy = 2
                else:
                    cjy = 4
                M = M - cjy * dx/3 * (sigma_c(curv*(xln-xj))*b*xj)
            return M

        #NEWTON-RAPHSON FUNCTION TO DETERMINE THE NEUTRAL AXIS POSITION
        def NR(deltax,x0,curv):
            xn = x0
            for j in range(0,max_iter):
                fxn = somaF(xn,curv)
                if abs(fxn) < epsilon:
                    return xn
                Dfxn = (somaF(xn+deltax,curv)-somaF(xn,curv))/deltax
                if Dfxn == 0:
                    return None
                xn = xn - fxn/Dfxn
            return None

        #INCREMENTAL PROCEDURE TO DETERMINE
        aini = 0.5*h    #initial guess for neutral axis position
        Ms = []
        Curves = []
        for k in range(1,max_iter):
            curva = dcurv*k
            a = NR(h/100,aini,curva)
            ecsup = curva * (a - h)
            ecinf = curva * (a - x1)
            if (ecsup < ecu) or (ecinf > esu):
                break
            M = somaM(a,curva)
            aini = a
            print(k,a,ecsup*1000,ecinf*1000,curva,M)
            Ms.append(M)
            Curves.append(curva)

        # PLOT GRAPHIC
        fig, ax = plt.subplots()
        ax.plot(Curves, Ms)
        title = 'Moment-curvature_relation'
        plt.title(r'Moment-curvature relation')
        plt.xlabel(r'Curvature')
        plt.ylabel(r'Moment')
        st.pyplot(fig)

        def savefigure():
            fig.savefig(title+'.eps', format='eps')
        action_save = st.button('Save as .eps', on_click=savefigure)

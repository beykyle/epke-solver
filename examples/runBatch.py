#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as mpl
import xml.etree.ElementTree as ET
from xml.dom import minidom
import subprocess

mpl.rcParams['font.size'] = 16


"""
This module runs epke in batch mode
"""

class Precursor:
    def __init__(self, decay_constant, beta):
        self.decay_constant = decay_constant
        self.beta = beta

class Data:
    def __init__(self,time,rho_impl,gen_time,pow_norm,rel_power,lambda_H,gamma_D):
        self.time = time
        self.rho_impl = rho_impl
        self.gen_time = gen_time
        self.pow_norm = pow_norm
        self.rel_power = rel_power
        self.lambda_H = lambda_H
        self.gamma_D = gamma_D

    def setBeff(self,precursors):
        self.beta_sum = sum([p.beta for p in precursors])
        self.beta_eff = self.beta_sum * np.ones(len(self.time))
        self.rho_impl *= self.beta_sum
        self.gamma_D *= self.beta_sum

    def write(self,precursors, fname):
        epke_input = ET.Element("epke_input", theta="0.5", gamma_d=str(self.gamma_D),
                        initial_power="1e-6", eta="1.0")
        precursors_element = ET.SubElement(epke_input, "precursors")

        for p in precursors:
            ET.SubElement(precursors_element, "precursor",
                          decay_constant=str(p.decay_constant),
                          beta=str(p.beta))
        ET.SubElement(epke_input, "time").text = ' '.join(map(str, self.time))
        ET.SubElement(epke_input, "gen_time").text = ' '.join(map(str, self.gen_time))
        ET.SubElement(epke_input, "pow_norm").text = ' '.join(map(str, self.pow_norm))
        ET.SubElement(epke_input, "rho_imp").text = ' '.join(map(str, self.rho_impl))
        ET.SubElement(epke_input, "beta_eff").text = ' '.join(map(str, self.beta_eff))
        ET.SubElement(epke_input, "lambda_h").text = ' '.join(map(str, self.lambda_H))
        # pretty print and write to xml
        xmlstr = minidom.parseString(ET.tostring(epke_input)).toprettyxml(indent="   ")
        with open(fname, "w") as f:
            f.write(xmlstr)

def loadData(data_fname, t_step):
    t_load, rho_imp_load, gen_time_load, pow_norm_load, \
    relative_power_load = np.loadtxt(data_fname,
                                     unpack=True,skiprows=2)

    t = np.linspace(0,max(t_load),int(max(t_load))*t_step+1)

    rho_imp_parcs = np.interp(t,t_load,rho_imp_load)

    rho_imp = np.copy(rho_imp_parcs)
    max_idx = np.argmax(rho_imp)
    rho_imp[max_idx:] = rho_imp[max_idx]

    gen_time = np.interp(t,t_load,gen_time_load)
    pow_norm = np.interp(t,t_load,pow_norm_load)
    relative_power = np.interp(t,t_load,relative_power_load)

    t = t[:6*t_step]
    rho_imp = rho_imp[:6*t_step]
    rho_imp_parcs = rho_imp_parcs[:6*t_step]
    gen_time = gen_time[:6*t_step]
    pow_norm = pow_norm[:6*t_step]
    relative_power = relative_power[:6*t_step]

    lambda_H = 0.29 * np.ones(len(t))
    gamma_D = -1.22

    return Data(t,rho_imp,gen_time,pow_norm,relative_power,lambda_H,gamma_D), rho_imp_parcs, relative_power

def runSolver(exec_fpath, fname):
    args = (exec_fpath + " " +  fname).split()
    popen = subprocess.Popen(args, cwd="..", stdout=subprocess.PIPE)
    popen.wait()
    #TODO add a timeout for failure
    while True:
        line = popen.stdout.readline()
        if not line:
            break
        print(line.rstrip().decode('utf-8'))

def extract_output(fname):
    tree = ET.parse('epke_output.xml')
    root = tree.getroot()

    for child in root:
        if (child.tag == 'time'):
            time_out = np.fromstring(child.attrib['values'], dtype=float, sep=',')
        if (child.tag == 'power'):
            p_out = np.fromstring(child.attrib['values'], dtype=float, sep=',')
        if (child.tag == 'rho'):
            rho_out = np.fromstring(child.attrib['values'], dtype=float, sep=',')

    return time_out, rho_out, p_out

def example():
    # set up a data set w/ 2 sets of precursor data
    precursors_a = [Precursor(0.0128, 0.02584e-2), Precursor(0.0318, 0.152e-2),
                  Precursor(0.119, 0.13908e-2), Precursor(0.3181, 0.30704e-2),
                  Precursor(1.4027, 0.1102e-2), Precursor(3.9286, 0.02584e-2)]
    precursor_data_sets = [ precursors_a]

    # read all the other data
    data, rho_PARCS, rel_power = loadData("cr_ejection_parcs.txt" ,  1000)

    # for each precursor data set, run the solver, extract and plot output
    for pset in precursor_data_sets:
        data.setBeff(pset)
        data.write(pset, "epke_input.xml")
        runSolver( "./epke-run", "examples/epke_input.xml")
        t, rho, p = extract_output("epke_ouput.xml")
        plt.plot(t,p)
        plt.plot(t,rel_power)
        plt.show()

if __name__ == "__main__":
    example()

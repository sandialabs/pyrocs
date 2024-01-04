import numpy as np
from metrics import cyclomatic_complexity, feedback_density, causal_complexity
from utils import load_ground_truth_graphML

# Segregation example
#ch_GTfile = "/Users/sjverzi/projects/Asmeret_Naugle/DARPA/GroundTruth/data/from_Vamshi/Schelling/ground_truth.graphml"
# Phase 1 models
#ch_GTfile = "/Users/sjverzi/projects/Asmeret_Naugle/DARPA/GroundTruth/challenges/phase 1/GMU/GroundTruth-GMU_revised09202018.graphml"
#ch_GTfile = "/Users/sjverzi/projects/Asmeret_Naugle/DARPA/GroundTruth/challenges/phase 1/Raytheon/GroundTruth-RAY-v4_fixed.graphml"
#ch_GTfile = "/Users/sjverzi/projects/Asmeret_Naugle/DARPA/GroundTruth/challenges/phase 1/USC/GroundTruth-USC.graphml"
#ch_GTfile = "/Users/sjverzi/projects/Asmeret_Naugle/DARPA/GroundTruth/challenges/phase 1/WSRI/WSRI Sim Eval Data Package/GroundTruth-WSR.graphml"
# Phase 2 models
#ch_GTfile = "/Users/sjverzi/projects/Asmeret_Naugle/DARPA/GroundTruth/challenges/phase 2/USC/USC/GroundTruth-USC.graphml"
ch_GTfile = "/Users/sjverzi/projects/Asmeret_Naugle/DARPA/GroundTruth/challenges/phase 2/WSRI/Files/SCAMP_GroundTruth.graphml"
G = load_ground_truth_graphML(ch_GTfile)

M = cyclomatic_complexity(G)
D = feedback_density(G)
C = causal_complexity(G)

print("cyclomatic complexity is " + str(M))
print("feedback density is " + str(D))
print("causal complexity is " + str(C))

from metrics import compute_NCD

for i in range(100, 1000, 100):
    #ncds = compute_NCD(np.zeros((100, i)))
    #print("IT for 0 avg is " + str(np.mean(ncds)) + " var is " + str(np.var(ncds)))
    #ncds = compute_NCD(np.ones((100, i)))
    #print("IT for 1 avg is " + str(np.mean(ncds)) + " var is " + str(np.var(ncds)))
    #ncds = compute_NCD(np.random.randint(0, 10, (100, i)))
    #print("IT for random avg is " + str(np.mean(ncds)) + " var is " + str(np.var(ncds)))
    #ncds = compute_NCD(np.sin(np.random.randint(0, 10, (100, i)) * np.pi / 20))
    #print("IT for sine of random avg is " + str(np.mean(ncds)) + " var is " + str(np.var(ncds)))
    #ncds = compute_NCD(np.array(range(i-100, i)))
    #print("IT for linear avg is " + str(np.mean(ncds)) + " var is " + str(np.var(ncds)))
    #ncds = compute_NCD(np.sin(np.array(range(i-100, i)) * np.pi / 10))
    #print("IT for sine of linear avg is " + str(np.mean(ncds)) + " var is " + str(np.var(ncds)))
    #ncds = compute_NCD(np.array(range(i)))
    #print("IT for linear avg is " + str(np.mean(ncds)) + " var is " + str(np.var(ncds)))
    ncds = compute_NCD(np.sin(np.array(range(i)) * np.pi / 10))
    print("IT for sine of linear avg is " + str(np.mean(ncds)) + " var is " + str(np.var(ncds)))

__author__ = 'dennis'

from PeakDensity import *
from sklearn.decomposition import PCA


def dist_dev(ins1, ins2):
    num_pos = range(len(ins1))
    str_pos = [i for i, x in enumerate(ins1.values) if type(x) == str]
    for x in str_pos:
        num_pos.remove(x)
    dis = np.dot(ins1[num_pos], ins1[num_pos]) + sum(ins1[str_pos] == ins2[str_pos])
    return np.sqrt(dis/2)

raw_data = []
ips = []
sep = [6, 10, 13, 18, 22, 26, 30]
labels = []

header = ['f01_total_unique_receivers', 'f02_total_unique_senders', 'f03_inter_receivers_senders',
          'f04_union_receivers_senders', 'f05_ratio_receivers_to_senders', 'f06_avg_freq_sending_pkg',
          'f07_avg_freq_receiving_pkg', 'f08_std_freq_sending_pkg', 'f09_std_freq_receiving_pkg',
          'f10_min_freq_sending_pkg', 'f11_q25_freq_sending_pkg', 'f12_q50_freq_sending_pkg',
          'f13_q75_freq_sending_pkg', 'f14_max_freq_sending_pkg', 'f15_min_freq_receiving_pkg',
          'f16_q25_freq_receiving_pkg', 'f17_q50_freq_receiving_pkg', 'f18_q75_freq_receiving_pkg',
          'f19_max_freq_receiving_pkg', 'f20_ratio_freq_send_receive_pkg', 'f21_corr_freq_send_receive_pkg',
          'f22_total_unique_prot_send', 'f23_total_unique_prot_receiced', 'f24_ratio_unique_prot_send_received',
          'f25_inter_prot_send_received', 'f26_union_prot_send_received', 'f27_total_unique_info_send',
          'f28_total_unique_info_receiced', 'f29_ratio_unique_info_send_received', 'f30_inter_info_send_received',
          'f31_union_info_send_received', 'f00_mac_manufacturer', 'f32_most_freq_send_prot',
          'f33_most_freq_received_prot']

with open('./Data/instances.txt', 'r') as f:
    for line in f:
        ip, ins = line.split(':')
        raw_data.append(dict(zip(header, ins.strip().split(', '))))
        ips.append(ip)

data = pd.DataFrame(raw_data, index=ips).convert_objects(convert_numeric=True)

for ip in ips:
    d = int(ip.split('.')[-1])
    labels.append([s for s, x in enumerate(sep) if x >= d][0])

n = len(data)
distance = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        distance[i, j] = dist_dev(data.iloc[i, :], data.iloc[j, :])
        distance[j, i] = distance[i, j]

mds = manifold.MDS(dissimilarity="precomputed")
coord = mds.fit(distance).embedding_

r, d, pk, cls = peak_density_cluster(distance, 0.002, gaussian=True)

pca = PCA(n_components=2)
pca.fit(data[header[0:31]])
X = pca.transform(data[header[0:31]])

results = pd.DataFrame({'rho': pd.Series(r), 'delta': pd.Series(d), 'peak': pd.Series(pk),
                        'cluster': pd.Series(cls), 'x': pd.Series(coord[:, 0]), 'y': pd.Series(coord[:, 1]),
                        'label': labels})

res2 = pd.DataFrame({'x': pd.Series(X[:, 0]), 'y': pd.Series(X[:, 1]), 'label': pd.Series(labels)})

g1 = ggplot(aes(x='rho', y='delta', color='peak'), data=results)+geom_point()
g1.draw()
g2 = ggplot(aes(x='x', y='y', color='cluster'), data=results)+geom_point()
print g2
# print ggplot(aes(x='x', y='y', color='label'), data=res2) + geom_point()

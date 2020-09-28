def dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

centroids = [(5,2), (5,4), (5,5)]
data = [(1,2), (1,3), (2,2), (2,3), (4,1), (4,2), (4,4), (4,5), (5,1), (5,2), (5,4), (5,5)]

def getAssignments():
    return [ min(range(len(centroids)), key=lambda c_i: dist(p1, centroids[c_i]) ) for p1 in data]

assignments = []

for epoch in range(100):
    print(epoch)
    old_assignments = assignments.copy()
    assignments = getAssignments()
    if old_assignments == assignments:
        break
    print(centroids)
    print(assignments)
    for k in range(len(centroids)):
        cluster = [data[i] for i, _ in filter(lambda x: x[1] == k, enumerate(assignments))]
        print(f"cluster {k}: {cluster}")
        if len(cluster) == 0:
            raise Exception(f"Uh oh for cluster {k}")
        centroids[k] = (0, 0)
        for c_data in cluster:
            centroids[k] = (centroids[k][0] + c_data[0], centroids[k][1] + c_data[1])
        centroids[k] = (centroids[k][0] / len(cluster), centroids[k][1] / len(cluster))
print(assignments)
print(centroids)

        
        # Creates empty list for the labels
        k_nearest_labels = []
        prediction_lab = []
        # Iterates the distance list from index 0 to the n'th index
        for i in distances[0:n]:
            # appends the label to the label list
            k_nearest_labels.append(i)
            #k_dist.append(i[0])

        zero = 0
        one = 0
        two = 0
        zero_acc_dist = 0
        one_acc_dist = 0
        two_acc_dist = 0
  
        for x in range (n):
            if k_nearest_labels[x][1] == 0:
                zero = zero + 1
                zero_acc_dist = zero_acc_dist + k_nearest_labels[x][0]
            elif k_nearest_labels[x][1] == 1:
                one = one + 1
                one_acc_dist = one_acc_dist + k_nearest_labels[x][0]
            elif k_nearest_labels[x][1] == 2:
                two = two + 1
                two_acc_dist = two_acc_dist + k_nearest_labels[x][0]
        
        print("occurences of zero", zero)
        print("occurences of one", one)
        print("occurences of two", two)
        #print("zero acc dist", zero_acc_dist)
        #print("one acc dist", one_acc_dist)
        #print("two acc dist", two_acc_dist)
        #print()
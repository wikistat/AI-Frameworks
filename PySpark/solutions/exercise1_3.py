attack_sum_count = attack_duration_data.aggregate(
    (0,0), # the initial value
    (lambda acc, value: (acc[0] + value, acc[1] + 1)), # combine value with acc
    (lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])) # combine accumulators
    )

print("Durée moyenne des interactions agressives {}".\
    format(round(attack_sum_count[0]/float(attack_sum_count[1]),3)))
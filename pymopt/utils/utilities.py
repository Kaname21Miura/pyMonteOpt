

def calTime(end, start):
    elapsed_time = end - start
    q, mod = divmod(elapsed_time, 60)
    if q < 60:
        print('Calculation time: %d minutes %0.3f seconds.' % (q, mod))
    else:
        q2, mod2 = divmod(q, 60)
        print('Calculation time: %d h %0.3f minutes.' % (q2, mod2))


def set_params(data,keys,*initial_data, **kwargs):
    for dictionary in initial_data:
        for key in dictionary:
            if not key in keys:
                raise KeyError(key)
            data[key] = dictionary[key]

    for key in kwargs:
        if not key in keys:
            raise KeyError(key)
        data[key] = kwargs[key]

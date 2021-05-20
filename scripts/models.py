

def main():
    # model 1: benchmark from 2020 Nat Comms paper
    from benchmark import model as model1
    model1()

    # model 2: model 1 with regional earthquakes filtered
    from eqfilter import model as model2
    model2()
    
    # model 3: model 2 with data standardisation
    from transformed import model as model3
    model3()

    # model 3.1: model 3 with probability calibration of output
    from probability import model as model3p
    model3p()
    
    # model 4: WSRZ model with model 3 filling gaps
    from uncertain_interpolation import model as model4
    model4()

    return

if __name__ == "__main__":
    main()
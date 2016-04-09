#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    
    ### your code goes here
    ### 'cleaned_data' is a list of tuples, where each tuple has the form (age, net_worth, error)

    from numpy import argmax, delete

    errors = (net_worths - predictions) ** 2
    clean_away = int(0.1 * len(errors))

    while clean_away > 0:
        index = argmax(errors)
        errors = delete(errors, index)
        ages = delete(ages, index)
        net_worths = delete(net_worths, index)
        clean_away -= 1

    cleaned_data = zip(ages, net_worths, errors)
                            
    return cleaned_data


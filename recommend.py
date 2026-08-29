def generate_recommendation(prediction):

    if prediction > 1000:
        return '''
        High Sales Demand Detected.

        Recommendation:
        - Increase inventory
        - Offer bundle discounts
        - Increase advertisement budget
        - Prepare warehouse stock
        '''

    elif prediction > 500:
        return '''
        Moderate Sales Demand.

        Recommendation:
        - Maintain current inventory
        - Use medium marketing campaigns
        - Monitor weekly sales trends
        '''

    else:
        return '''
        Low Sales Demand.

        Recommendation:
        - Reduce overstock risk
        - Offer promotional discounts
        - Focus on customer engagement
        '''
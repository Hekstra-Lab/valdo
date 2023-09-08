def print_df_with_lists(df,rows=None, cols_to_format):
    def float2string(input):
        """convert float to string for printing
    
        Input (float)
        Output (string)
    
        Thanks, StackOverflow!
        """
        if hasattr(input, '__iter__'):
            return list(map(float2string, input))
        else:
            if input is None:
                return None
            else:
                if float(input).is_integer():
                    return "{}".format(input)
                if abs(input) < 1e-2 or abs(input) > 1e2:
                    return "{:.2e}".format(input)
                else:
                    return "{:.3f}".format(input)
    formatters={}
    for col in cols_to_format:
        formatters[col]=float2string
    if rows is None:            
        print(df.loc[:,cols].to_string(float_format = float2string, formatters = formatters))
    else:
        print(df.loc[rows,cols].to_string(float_format = float2string, formatters = formatters))
    return 0
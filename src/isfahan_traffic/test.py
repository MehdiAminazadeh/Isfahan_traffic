from isfahan_traffic.parser import parse_timestamp

messy_string = """
Thursday 01 November 2018 00:15
Int 1081 1=14 2=51 3=38
"""

first_line = messy_string.strip().splitlines()[0]

timestamp = parse_timestamp(first_line)

print(timestamp)
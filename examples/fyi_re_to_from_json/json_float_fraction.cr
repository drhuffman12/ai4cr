require "json"
# struct Float64
#   include ::JSON::Serializable
# end

# Test what max values can be to/from-jsonified
(0..309).each do |exp|
  puts "*"*20
  p! exp
  f = 0.0
  p!(f = 10.0**(-exp))
  p! f
  j = f.to_json
  p! j
  f2 = Float64.from_json(j)
  p! f2
end

# This outputs:
# ...
# ********************
# exp # => 307
# f = 10.0 ** (-exp) # => 9.999999999999995e-308
# f # => 9.999999999999995e-308
# j # => "9.999999999999995e-308"
# f2 # => 9.999999999999991e-308
# ********************
# exp # => 308
# f = 10.0 ** (-exp) # => 9.999999999999994e-309
# f # => 9.999999999999994e-309
# j # => "9.999999999999994e-309"
# f2 # => 0.0
# ********************
# exp # => 309
# f = 10.0 ** (-exp) # => 0.0
# f # => 0.0
# j # => "0.0"
# f2 # => 0.0

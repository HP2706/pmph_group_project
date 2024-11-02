import "lib/github.com/diku-dk/sorts/radix_sort"


let sort_u16 = radix_sort u16.num_bits u16.get_bit

-- this is so ugly 
let main (arr: []u16) : []u16 = 
  sort_u16 arr

PATTERN = "{ \"name\" : \"Neg_%s\",\n" \
          "\t\t\"type\" : \"neg\",\n" \
          "\t\t\"host\" : \"data/simulated/null/null%s/host.tree\",\n " \
          "\t\t\"guest\": \"data/simulated/null/null%s/guest.tree\",\n" \
          "\t\t\"links\": \"data/simulated/null/null%s/links.csv\" }"
def gen_negjson():
    fout = open("../jsinfo/negative_pairs.json", "w")
    fout.write("[\n")
    for i in range(50):
        fout.write(PATTERN % (i,i,i,i))
        if i == 49:
            fout.write("\n\n]")
        else:
            fout.write(",\n")
    fout.close()
if __name__ == "__main__":
    gen_negjson()
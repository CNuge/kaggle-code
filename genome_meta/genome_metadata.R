####
# TO DO
# 2. grab some fun summary facts on the outlier genomes, which species are analmolous?
# 4. Further dive in to the genomes of eukaryotes, 
		#split by organism groups
		# compare sizes across the land and aquatic plants
		# compare vertebrates to plants
######




library('tidyverse')


raw_eukaryote = read_csv('all_genomes_lists/eukaryotes.csv')
names(raw_eukaryote)
raw_eukaryote


#mean + median
mean(raw_eukaryote$"GC%")
median(raw_eukaryote$"GC%")


# just the chromosome and scaffold level assembies, the contigs removed

eukaryote = raw_eukaryote[raw_eukaryote$Level == "Chromosome" || raw_eukaryote$Level == "Scaffold",]
eukaryote

eukaryote = eukaryote[eukaryote$"GC%" > 5,]

#Prototheca stagnorum has 71% GC content which is pretty neat
eukaryote[eukaryote$"GC%" > 70,]

#mean + median
mean(eukaryote$"GC%")
median(eukaryote$"GC%")

#looking at the data there appears to be some missing data.
# GC of zero seen, looks like one with with a very low number,
# also an outlier with >70% gc, I should look at what this one is 
qplot(eukaryote$"GC%")



prokaryote = read_csv('all_genomes_lists/prokaryotes.csv')
names(prokaryote)

prokaryote = prokaryote[prokaryote$"GC%" > 5,]

#check to make sure things aren't valuable
low_prokaryote = prokaryote[prokaryote$"GC%" < 20,]
low_prokaryote
as.data.frame(low_prokaryote)

#wider distribution for prokaryotes
# higher mean and median
qplot(prokaryote$"GC%")

mean(prokaryote$"GC%")
median(prokaryote$"GC%")


#one 0 in the viruses... also a gc as high as 80+ investigate further
viruses = read_csv('all_genomes_lists/viruses.csv')
names(viruses)
viruses = viruses[viruses$"GC%" > 5,]


qplot(viruses$"GC%")

#mean of this GC in between the two others
mean(viruses$"GC%")
median(viruses$"GC%")



boxplot(list(viruses = viruses$"GC%", 
			prokaryote = prokaryote$"GC%", 
			eukaryote = eukaryote$"GC%"))

all_genomes = data.frame(kingdom= "virus", 
						GC = viruses$"GC%", 
						size_mb = viruses$"Size(Mb)" )

all_genomes = rbind(all_genomes, data.frame(kingdom="prokaryote", GC = prokaryote$"GC%", size_mb = prokaryote$"Size(Mb)"))
all_genomes = rbind(all_genomes, data.frame(kingdom="eukaryote", GC = eukaryote$"GC%", size_mb = eukaryote$"Size(Mb)"))

head(all_genomes)
tail(all_genomes)


gc_anova = aov(GC ~ kingdom , data=all_genomes)
summary(gc_anova)

TukeyHSD(gc_anova)


#differences in size? should be obvious
mb_anova = aov(size_mb ~ kingdom , data=all_genomes)
summary(mb_anova)
TukeyHSD(mb_anova)
# the virus/prokaryotes not sig different, suggestive of prokaryote fragments in the data


ggplot(all_genomes, aes(x=size_mb, y=GC, shape=kingdom, colour=kingdom)) +
        geom_point() +
        scale_colour_brewer(palette="Set1") +
        coord_trans(x = "log10") +
        labs(title = "GC content vs genome size (mb) of genome sequence stored on NCBI", 
        	y = "%GC content", x = "Genome size (mb)")








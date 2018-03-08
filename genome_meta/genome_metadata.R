####
# TO DO
# 2. grab some fun summary facts on the outlier genomes, which species are analmolous?
# 4. Further dive in to the genomes of eukaryotes, 
		#split by organism groups
		# compare sizes across the land and aquatic plants
		# compare vertebrates to plants
######

library('tidyverse')
library('ggthemes')

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

all_genomes = rbind(all_genomes, data.frame(kingdom="prokaryote", 
												GC = prokaryote$"GC%", 
												size_mb = prokaryote$"Size(Mb)"))
all_genomes = rbind(all_genomes, data.frame(kingdom="eukaryote", 
												GC = eukaryote$"GC%", 
												size_mb = eukaryote$"Size(Mb)"))
	
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
        labs(title = "GC content vs genome size (mb) of genome sequences stored on NCBI", 
        		y = "%GC content", 
        		x = "Genome size (mb) - Note this axis is log scaled") +
        theme(axis.text.x = element_text(angle = 90, hjust = 1))



#plot of the gc content by kingdom
ggplot(all_genomes, aes(x=kingdom, y=GC, fill=kingdom, color=kingdom))+
	geom_boxplot(outlier.shape = NA) +
	geom_point(shape = 21, alpha = 0.1, position=position_jitterdodge())+
	scale_fill_manual(values = rep(NA,3)) +
	theme_fivethirtyeight() +
	scale_colour_economist()



#######
## load source code
#locally:
# source("geom_flat_violin.R")

## sourced from github "dgrtwo/geom_flat_violin.R
## https://gist.github.com/dgrtwo/eb7750e74997891d7c20#file-geom_flat_violin-r

# I have pasted in the raw function below for the use in this kernel
#######


"%||%" <- function(a, b) {
  if (!is.null(a)) a else b
}

geom_flat_violin <- function(mapping = NULL, data = NULL, stat = "ydensity",
                        position = "dodge", trim = TRUE, scale = "area",
                        show.legend = NA, inherit.aes = TRUE, ...) {
  layer(
    data = data,
    mapping = mapping,
    stat = stat,
    geom = GeomFlatViolin,
    position = position,
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(
      trim = trim,
      scale = scale,
      ...
    )
  )
}


#' @rdname ggplot2-ggproto
#' @format NULL
#' @usage NULL
#' @export
GeomFlatViolin <-
  ggproto("GeomFlatViolin", Geom,
          setup_data = function(data, params) {
            data$width <- data$width %||%
              params$width %||% (resolution(data$x, FALSE) * 0.9)
            
            # ymin, ymax, xmin, and xmax define the bounding rectangle for each group
            data %>%
              group_by(group) %>%
              mutate(ymin = min(y),
                     ymax = max(y),
                     xmin = x,
                     xmax = x + width / 2)
          },
          
          draw_group = function(data, panel_scales, coord) {
            # Find the points for the line to go all the way around
            data <- transform(data, xminv = x,
                              xmaxv = x - violinwidth * (xmax - x)) # change x - violinwidth to + for violin to right
            
            # Make sure it's sorted properly to draw the outline
            newdata <- rbind(plyr::arrange(transform(data, x = xminv), y),
                             plyr::arrange(transform(data, x = xmaxv), -y))
            
            # Close the polygon: set first and last point the same
            # Needed for coord_polar and such
            newdata <- rbind(newdata, newdata[1,])
            
            ggplot2:::ggname("geom_flat_violin", GeomPolygon$draw_panel(newdata, panel_scales, coord))
          },
          
          draw_key = draw_key_polygon,
          
          default_aes = aes(weight = 1, colour = "grey20", fill = "white", size = 0.5,
                            alpha = NA, linetype = "solid"),
          
          required_aes = c("x", "y")
)


#####
# Violin + scatter plot of the gc content of genomes across different organism groups
#####

ggplot(all_genomes, aes(x = kingdom, y = GC, fill=kingdom)) +
	geom_flat_violin(colour="white") +
	geom_point(aes(x = as.numeric(kingdom) + .12, colour=kingdom) , 
					size = 0.01, 
          			alpha = 0.1,
					pch=21, 
					position = position_jitterdodge(jitter.width = .55, jitter.height = 0.01, dodge.width = 0.75)) +
	theme_fivethirtyeight() +
    labs(title = "GC content of genome sequences stored on NCBI", 
        	y = "%GC content", 
        	x = "Genome size (mb)")



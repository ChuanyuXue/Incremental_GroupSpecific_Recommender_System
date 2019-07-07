library(bipartite)


edge2web=function(sample)
{


  x_id = sort(unique(sample[,1]))
  y_id = sort(unique(sample[,2]))

  web = matrix(0,nrow = length(x_id), ncol = length(y_id))

  rownames(web) = as.character(x_id)
  colnames(web) = as.character(y_id)

  for(i in 1:nrow(sample))
  {
    web[sample[i,1],sample[i,2]] = sample[i,3]
  }

  return(web)
}


edges = read.table('C:/Users/XueChuanyu/Desktop/git_gsrs/Group-Specific-Recommender-System/data/BX_data.csv',sep = ',')

web = edge2web(edges)
res <- computeModules(web)
res@likelihood


moduleList = listModuleInformation(res)

printoutModuleInformation(res)





function pngprint(name)
set(gcf,'PaperPositionMode','auto')
print(gcf,'-dpng','-r300',[name '.png'])
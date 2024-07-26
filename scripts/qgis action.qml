<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="Actions" version="3.28.3-Firenze">
  <attributeactions>
    <defaultAction key="Canvas" value="{edaf6d57-9177-42b7-af69-1bfb79fe2171}"/>
    <actionsetting notificationMessage="" shortTitle="Open Bruty export" id="{edaf6d57-9177-42b7-af69-1bfb79fe2171}" name="Open Bruty export" icon="" capture="0" type="1" action="import os&#xd;&#xa;from qgis.core import QgsRasterLayer&#xd;&#xa;from qgis.utils import iface&#xd;&#xa;import glob&#xd;&#xa;from qgis.core import QgsVectorLayer, QgsProject&#xd;&#xa;active = iface.activeLayer()&#xd;&#xa;grp_name = r&quot;Bruty &quot;+active.name()+&quot; Exports&quot;&#xd;&#xa;exist = False&#xd;&#xa;root = QgsProject.instance().layerTreeRoot()&#xd;&#xa;for rgroup in root.findGroups():&#xd;&#xa;    if rgroup.name() == grp_name:&#xd;&#xa;        exist = True&#xd;&#xa;        group = rgroup&#xd;&#xa;        break&#xd;&#xa;if not exist:&#xd;&#xa;    group = root.insertGroup(0, grp_name)&#xd;&#xa;bag = r&quot;[%data_location%]&quot;&#xd;&#xa;base = os.path.basename(bag)&#xd;&#xa;layers = QgsProject.instance().mapLayersByName(base)&#xd;&#xa;if not layers:&#xd;&#xa;    if os.path.isfile(bag):&#xd;&#xa;        lyr = QgsRasterLayer(bag, base)&#xd;&#xa;        QgsProject.instance().addMapLayer(lyr, False)&#xd;&#xa;        group.addLayer(lyr)" isEnabledOnlyWhenEditable="0">
      <actionScope id="Feature"/>
      <actionScope id="Canvas"/>
    </actionsetting>
  </attributeactions>
  <layerGeometryType>2</layerGeometryType>
</qgis>

package org.explv.mapimage;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;
import java.util.function.BiConsumer;
import javax.imageio.ImageIO;
import lombok.extern.slf4j.Slf4j;
import net.runelite.cache.MapImageDumper;
import net.runelite.cache.fs.Store;
import net.runelite.cache.region.Region;
import net.runelite.cache.util.XteaKeyManager;
import org.antlr.v4.runtime.misc.Pair;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

@Slf4j
public class Main {
	private static final List<Pair<String, BiConsumer<MapImageDumper, Boolean>>> mapOptions = List.of(
		new Pair<>("renderMap", MapImageDumper::setRenderMap),
		new Pair<>("renderObjects", MapImageDumper::setRenderObjects),
		new Pair<>("renderIcons", MapImageDumper::setRenderIcons),
		new Pair<>("renderWalls", MapImageDumper::setRenderWalls),
		new Pair<>("renderOverlays", MapImageDumper::setRenderOverlays),
		new Pair<>("renderLabels", MapImageDumper::setRenderLabels),
		new Pair<>("transparency", MapImageDumper::setTransparency)
	);

	public static void main(String[] args) throws IOException {
		Options options = new Options();
		options.addOption(Option.builder().longOpt("cachedir").hasArg().required().build());
		options.addOption(Option.builder().longOpt("xteapath").hasArg().required().build());
		options.addOption(Option.builder().longOpt("outputdir").hasArg().required().build());

		// read in custom render options, runelite doesn't let you set these by default
		for (Pair<String, BiConsumer<MapImageDumper, Boolean>> mapOption : mapOptions) {
			options.addOption(Option.builder().longOpt(mapOption.a).hasArg().build());
		}

		CommandLineParser parser = new DefaultParser();
		CommandLine cmd;
		try
		{
			cmd = parser.parse(options, args);
		}
		catch (ParseException ex)
		{
			System.err.println("Error parsing command line options: " + ex.getMessage());
			System.exit(-1);
			return;
		}

		final String cacheDirectory = cmd.getOptionValue("cachedir");
		final String xteaJSONPath = cmd.getOptionValue("xteapath");
		final String outputDirectory = cmd.getOptionValue("outputdir");

		XteaKeyManager xteaKeyManager = new XteaKeyManager();
		try (FileInputStream fin = new FileInputStream(xteaJSONPath))
		{
			xteaKeyManager.loadKeys(fin);
		}

		File base = new File(cacheDirectory);
		File outDir = new File(outputDirectory);
		outDir.mkdirs();

		try (Store store = new Store(base))
		{
			store.load();

			MapImageDumper dumper = new MapImageDumper(store, xteaKeyManager);

			// apply custom render options
			for (Pair<String, BiConsumer<MapImageDumper, Boolean>> mapOption : mapOptions) {
				if (cmd.hasOption(mapOption.a)) {
					String option = cmd.getOptionValue(mapOption.a);
					if (option.equalsIgnoreCase("true")) {
						mapOption.b.accept(dumper, true);
					} else {
						mapOption.b.accept(dumper, false);
					}
				}
			}

			dumper.load();

			for (int i = 0; i < Region.Z; ++i)
			{
				BufferedImage image = dumper.drawMap(i);

				File imageFile = new File(outDir, "img-" + i + ".png");

				ImageIO.write(image, "png", imageFile);
				log.info("Wrote image {}", imageFile);
			}
		}
	}
}
